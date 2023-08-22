import re
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import lightgbm as lgb
from itertools import chain
import datetime
from tqdm import tqdm
from config import *


def get_rm_daily_flag(data):
	data["sales_date"] = pd.to_datetime(data["sales_date"])
	rm_calendar = pd.read_excel(os.path.join(paths.source, rm_calendar_src))
	return pd.merge(data, rm_calendar[["sales_date", "is_rm", "is_preheat"]], how="left")


def get_tmall_traffic(traffic):
	traffic.columns = ["platform", "sales_date", "site_traffic"]
	tmall_traffic_sub_platform = traffic[traffic["platform"].isin(['TMALL Jordan', 'Tmall', 'TMALL YA'])]
	tmall_traffic = tmall_traffic_sub_platform.groupby(["sales_date"]).sum()
	return tmall_traffic


def cal_sku_num_in_style(ftw_src, time_scope="season"):
	"""
		time_scope: season("sn"):这个season里这个style有几个sku
					sales_date("dt"): 今天这个style里有几个sku
	"""
	time_scope_use_col = "season1" if time_scope == "season" else "sales_date"
	time_scope_col = "sn" if time_scope == "season" else "dt"
	sn_style_sku_num = ftw_src.groupby([time_scope_use_col, "style_cd"]).agg({"prod_code": pd.Series.nunique})
	sn_style_sku_num.rename(columns={"prod_code": f"same_style_sku_num_{time_scope_col}"}, inplace=True)
	return sn_style_sku_num


def sku_to_style_attribute_transform(ftw_src):
	numerical_cols = ["sales_qty", "sales_amt", "inv_qty", "md"]
	qty_attr = ftw_src.groupby(["sales_date", "style_cd", "season1", "inseason_flag1"]).agg({
		"sales_qty": sum,
		"sales_amt": sum,
		"inv_qty": sum,
		"md": 'mean'}).reset_index()
	temp = ftw_src.sort_values(by=["sales_date", "season1", "inseason_flag1", "style_cd", "sales_qty"], ascending=False)
	temp = temp.drop(numerical_cols, axis=1)
	style_attr = temp.drop_duplicates(subset=["sales_date", "season1", "inseason_flag1", "style_cd"], keep="first")
	style_base = pd.merge(qty_attr, style_attr, on=["sales_date", "season1", "inseason_flag1", "style_cd"], how="left")
	return style_base


def get_past_x_day_tot_sales_qty(ftw_model_base, rm_remove=True, past_days=14, type='min', shift_days=1,
								 level="prod_code", keep_full_price=True, drop_md=False):
	ftw_model_base["sales_date"] = pd.to_datetime(ftw_model_base["sales_date"])
	if rm_remove:
		hist_md_aps_rm_removed = ftw_model_base[
			~ftw_model_base["sales_date"].isin([pd.to_datetime(x) for x in rm_dates])]
	else:
		hist_md_aps_rm_removed = ftw_model_base.copy()
	if not keep_full_price:
		hist_md_aps_rm_removed = hist_md_aps_rm_removed[round(hist_md_aps_rm_removed["md"]) <= 0.9]  # remove full price
	if drop_md:
		hist_md_aps_rm_removed = hist_md_aps_rm_removed[
			round(hist_md_aps_rm_removed["md"]) > 0.9]  # keep only full price
	prod_daily_sales = pd.crosstab(index=hist_md_aps_rm_removed["sales_date"], columns=hist_md_aps_rm_removed[level],
								   values=hist_md_aps_rm_removed["sales_qty"], aggfunc='mean')
	# past_14d_full_price_aps = eval(f"prod_daily_sales.rolling('{past_days}').{agg_func}().shift(1)")
	if type == 'avg':
		past_14d_full_price_aps = prod_daily_sales.rolling(past_days).mean().shift(shift_days, freq='D')
	elif type == 'max':
		past_14d_full_price_aps = prod_daily_sales.rolling(past_days).max().shift(shift_days, freq='D')
	elif type == 'min':
		past_14d_full_price_aps = prod_daily_sales.rolling(past_days).min().shift(shift_days, freq='D')
	elif type == 'median':
		past_14d_full_price_aps = prod_daily_sales.rolling(past_days).median().shift(shift_days, freq='D')
	elif type == 'std':
		past_14d_full_price_aps = prod_daily_sales.rolling(past_days).std().shift(shift_days, freq='D')
	elif type == 'sum':
		past_14d_full_price_aps = prod_daily_sales.rolling(past_days).sum().shift(shift_days, freq='D')
	elif type == 'count':
		past_14d_full_price_aps = prod_daily_sales.rolling(past_days).count().shift(shift_days, freq='D')
	else:
		raise ValueError("wrong type")
	past_14d_full_price_aps_stack = past_14d_full_price_aps.ffill().stack()
	md_type = "" if keep_full_price else "md"
	full_price = "full_price_" if drop_md else ""
	past_14d_full_price_aps_stack.name = f"past_{past_days}_{full_price}{type}_{md_type}aps"
	return past_14d_full_price_aps_stack


def cal_md_round(data, level="style_cd"):
	# 计算是第几轮打折，这轮打折的第几天
	grp_index = [level, "season1", "inseason_flag1"]
	a = data.copy()
	a["sales_date"] = pd.to_datetime(a["sales_date"])
	a = a.sort_values(by=[level, "sales_date"], ascending=True)
	b = a.query("md < 1")
	c = b.groupby(grp_index).apply(lambda x: x["md"].diff()).reset_index().set_index("level_3")  # 计算每个产品每天相较于前一天的折扣差
	c["is_new_round"] = abs(c["md"]) > 0  # 是否是新一轮打折
	c["md_diff"] = np.where(c["is_new_round"] > 0, c["md"], 0)  # 这轮打折和上轮打折的折扣差
	cc = pd.merge(a["sales_date"], c, left_index=True, right_index=True, how="right")

	grp_index.append("sales_date")
	cum_round = cc.groupby(grp_index)["is_new_round"].sum().groupby(level=[0, 1, 2]).cumsum()
	cum_round.name = "cum_round"  # 第几轮打折
	d = pd.merge(cc.reset_index(), cum_round, on=cum_round.index.names, how="left")
	d["is_new_round"] = ~d["is_new_round"]
	round_days = d.groupby([level, "season1", "inseason_flag1", "cum_round", "sales_date"])[
		"is_new_round"].sum().groupby(level=[0, 1, 2, 3]).cumsum()
	round_days.name = "round_days"
	e = pd.merge(d, round_days, on=round_days.index.names, how="left")  # 这轮打折的第几天
	# f = e[[level, "inseason_flag1", "sales_date","cum_round", "is_new_round", "round_days"]]
	e = e[[level, "inseason_flag1", "sales_date", "round_days", "md_diff", "cum_round"]]
	e = e.set_index([level, "inseason_flag1", "sales_date"])
	return pd.merge(data, e, on=e.index.names, how="left")


def simpliflied_color(x, pos=0):
	color_list = pd.read_excel(f"{paths.source}/{cleaned_color_words}")["color"].tolist()
	try:
		color_desc = x.split("/")
		base_color = color_desc[pos]
		color_words = re.split("[- ]", base_color)
		if len(color_words) > 1:
			color_words = [x for x in color_words if x in color_list]
			color_words = list(set(color_words))
			color_words.sort()
			return " ".join(color_words)
	except:
		print(x)
	return x


def get_major_minor_color_features(data):
	data["base_color"] = data["prod_code"].str[-3:-2]
	data["Full Color Description"] = data["Full Color Description"].str.lower()
	data["major_color_desc"] = data["Full Color Description"].apply(
		lambda x: simpliflied_color(x, 0))
	data["minor_color_desc"] = data["Full Color Description"].apply(
		lambda x: simpliflied_color(x, 1))
	return data[["sales_date", level_col, "base_color", "major_color_desc", "minor_color_desc"]]


def extract_colors(data):
	print("extract major and minor color features")
	try:
		old_data = pd.read_csv(os.path.join(paths.step_data, sku_color_features_to_date))
		if fcst_date:
			old_data = old_data.query("sales_date != @fcst_date")
		data_to_date = old_data.sales_date.max()
		old_features = old_data[
			["sales_date", "prod_code", "base_color", "major_color_desc", "minor_color_desc"]]
		new_data = data.query("sales_date > @data_to_date")
		new_features = get_major_minor_color_features(new_data)
		color_features_tot = pd.concat([old_features, new_features])
	except:
		color_features_tot = get_major_minor_color_features(data)
	color_features_tot["sales_date"] = pd.to_datetime(color_features_tot["sales_date"])
	color_features_tot.to_csv(os.path.join(paths.step_data, sku_color_features_to_date), index=False)
	return pd.merge(data, color_features_tot, on=["sales_date", "prod_code"], how="left")


class FeatureMaker:

	def __init__(self, data, level="sku", rm_remove=True):
		self.data = data.copy()
		self.level = "style_cd" if level == "style" else "prod_code"
		self.rm_remove = rm_remove
		self.data["md"] = self.data["md"].round(1)
		self.data["sales_date"] = pd.to_datetime(self.data["sales_date"])

	# def extract_color(self):
	# print("extract major and minor color features")
	# self.data["base_color"] = self.data["prod_code"].str[-3:-2]
	# self.data["Full Color Description"] = self.data["Full Color Description"].str.lower()
	# self.data["major_color_desc"] = self.data["Full Color Description"].apply(
	# 	lambda x: simpliflied_color(x, 0))
	# self.data["minor_color_desc"] = self.data["Full Color Description"].apply(
	# 	lambda x: simpliflied_color(x, 1))

	def get_same_md_hist_cum_max_min_aps(self, type='max', shift_days=0):
		if self.rm_remove:
			hist_md_aps_rm_removed = self.data[~self.data["sales_date"].isin([pd.to_datetime(x) for x in rm_dates])]
		else:
			hist_md_aps_rm_removed = self.data.copy()
		if type == 'max':
			hist_max_aps = pd.crosstab(index=hist_md_aps_rm_removed["sales_date"],
									   columns=[hist_md_aps_rm_removed[self.level], hist_md_aps_rm_removed["md"]],
									   values=hist_md_aps_rm_removed["sales_qty"], aggfunc='mean').cummax().shift(
				shift_days, freq='D')
		elif type == 'min':
			hist_max_aps = pd.crosstab(index=hist_md_aps_rm_removed["sales_date"],
									   columns=[hist_md_aps_rm_removed[self.level], hist_md_aps_rm_removed["md"]],
									   values=hist_md_aps_rm_removed["sales_qty"], aggfunc='mean').cummin().shift(
				shift_days, freq='D')
		elif type == 'mean':
			hist_sum_aps = pd.crosstab(index=hist_md_aps_rm_removed["sales_date"],
									   columns=[hist_md_aps_rm_removed[self.level], hist_md_aps_rm_removed["md"]],
									   values=hist_md_aps_rm_removed["sales_qty"], aggfunc='mean').cumsum().shift(
				shift_days, freq='D')
			hist_count_aps = pd.crosstab(index=hist_md_aps_rm_removed["sales_date"],
										 columns=[hist_md_aps_rm_removed[self.level],
												  hist_md_aps_rm_removed["md"]]).cumsum().shift(shift_days, freq='D')
			hist_max_aps = hist_sum_aps / hist_count_aps
		else:
			raise ValueError("type should be one of max/min/mean")
		hist_max_aps = hist_max_aps.ffill().stack().stack()
		hist_max_aps.name = f"same_md_hist_{type}_aps"
		hist_max_aps = hist_max_aps.reset_index()
		return hist_max_aps

	def same_md_hist_frequency(self):
		print("same_md_hist_frequency")
		# 每个产品截止到每一天，打到这个折扣的次数(包含当天)
		md_frequency_by_date = pd.crosstab(self.data["sales_date"], [self.data[self.level], self.data["md"]]).cumsum()
		self.hist_md_frequency = md_frequency_by_date.stack().stack()
		self.hist_md_frequency.name = "same_md_hist_frequency"
		self.data = pd.merge(self.data, self.hist_md_frequency, left_on=["sales_date", "md", self.level],
							 right_index=True, how="left")

	def inseason_same_md_hist_frequency(self):
		print("inseason_same_md_hist_frequency")
		# 本次inseason打到这个折扣的次数
		calendar = pd.read_excel(f"{paths.source}/{calendar_file}", usecols=["DATE", "Season"])
		calendar.rename(columns={"DATE": "sales_date", "Season": "season1"}, inplace=True)
		hist_md_frequency_df = self.hist_md_frequency.reset_index()
		hist_md_frequency_df["sales_date"] = pd.to_datetime(hist_md_frequency_df["sales_date"])
		hist_md_frequency_df = pd.merge(hist_md_frequency_df, calendar, left_on="sales_date", right_on="sales_date",
										how="left")
		hist_md_frequency_df["season_rank"] = hist_md_frequency_df["season1"].replace(
			dict(zip(season_list, range(len(season_list)))))
		last_season_md_frequency = hist_md_frequency_df.drop_duplicates(subset=["season1", self.level, "md"],
																		keep='last')
		last_season_md_frequency["season_rank"] = last_season_md_frequency["season_rank"] + 1
		last_season_md_frequency.rename(columns={"same_md_hist_frequency": "same_md_inseason_freq"}, inplace=True)
		# merge [same_md_inseason_freq] to master
		self.data["season_rank"] = self.data["season1"].replace(dict(zip(season_list, range(len(season_list)))))
		self.data = pd.merge(self.data,
							 last_season_md_frequency[[self.level, "md", "season_rank", "same_md_inseason_freq"]],
							 on=[self.level, "md", "season_rank"], how="left")
		self.data["same_md_inseason_freq"] = self.data.eval("same_md_hist_frequency - same_md_inseason_freq")

	def hist_same_md_aps_stats(self, shift_days=14):
		print("hist_same_md_aps_stats")
		# merge [same_md_hist_min_aps], [same_md_hist_max_aps], [same_md_hist_mean_aps] to master
		for type in ["max", "min", "mean"]:
			hist_aps = self.get_same_md_hist_cum_max_min_aps(type=type, shift_days=shift_days)
			self.data = pd.merge(self.data, hist_aps, on=["sales_date", "md", self.level], how="left")

	def cal_inseason_this_md_previous_aps(self, shift_days):
		print("cal_inseason_this_md_previous_aps")
		if self.rm_remove:
			use_data = self.data[~self.data["sales_date"].isin([pd.to_datetime(x) for x in rm_dates])]
		else:
			use_data = self.data.copy()
		new_col_df = pd.DataFrame()
		for season in use_data["season1"].unique():
			season_data = use_data.query("season1 == @season")
			md_qty = pd.crosstab(index=season_data["sales_date"], columns=[season_data[self.level], season_data["md"]],
								 values=season_data["sales_qty"], aggfunc='sum').cumsum().ffill()
			md_count = pd.crosstab(index=season_data["sales_date"],
								   columns=[season_data[self.level], season_data["md"]]).cumsum().ffill()
			md_aps = md_qty / md_count
			inseason_previous_md_aps = md_aps.shift(shift_days, freq='D')
			inseason_previous_md_aps = inseason_previous_md_aps.ffill().stack().stack()
			inseason_previous_md_aps.name = "inseason_previous_md_aps"
			new_col_df = pd.concat([new_col_df, inseason_previous_md_aps.reset_index()])
		self.data = pd.merge(self.data, new_col_df, on=["sales_date", self.level, "md"], how="left")

	def cal_hist_md_avg(self):
		print("cal_hist_md_avg")
		# 历史的平均每日md，不算今日
		hist_md = pd.crosstab(index=self.data["sales_date"], columns=self.data[self.level], values=self.data["md"],
							  aggfunc='mean')
		hist_md_avg = (hist_md.cumsum() / hist_md.notnull().cumsum()).shift(1)
		hist_md_avg = hist_md_avg.ffill().stack()
		hist_md_avg.name = "hist_md_avg"
		self.data = pd.merge(self.data, hist_md_avg, left_on=hist_md_avg.index.names, right_index=True, how="left")

	def cal_all_md_prior_7d_aps(self, shift_days=14):
		print("cal_all_md_prior_7d_aps")
		# 计算14天前的滚动aps（full price计算在内）
		for rolling_days in [1, 3, 7, 14, 21, 28]:
			for type in ["max", "min", "std", "median", "avg"]:
				if (rolling_days == 1) & (type != 'avg'):
					continue
				past = get_past_x_day_tot_sales_qty(self.data, rm_remove=self.rm_remove, past_days=rolling_days,
													type=type, shift_days=shift_days, level=self.level,
													keep_full_price=True)
				self.data = pd.merge(self.data, past, left_on=past.index.names, right_index=True, how="left")

	def cal_md_prior_7d_aps(self, shift_days=14):
		print("cal_md_prior_7d_aps")
		# 14天前的滚动aps（full price不计算在内）
		for rolling_days in [2, 7, 14]:
			for type in ["max", "min", "std", "avg"]:
				past = get_past_x_day_tot_sales_qty(self.data, rm_remove=self.rm_remove, past_days=rolling_days,
													type=type, shift_days=shift_days, level=self.level,
													keep_full_price=False)
				self.data = pd.merge(self.data, past, left_on=past.index.names, right_index=True, how="left")

	def cal_prior_7d_full_price_aps(self, shift_days=14):
		print("cal_prior_7d_full_price_aps")
		# 14天前的滚动aps（full price----折扣力度小于9折的aps）
		for rolling_days in [7, 14]:
			for type in ["std", "avg"]:
				past = get_past_x_day_tot_sales_qty(self.data, rm_remove=self.rm_remove, past_days=rolling_days,
													type=type, shift_days=shift_days, level=self.level,
													keep_full_price=True, drop_md=True)
				self.data = pd.merge(self.data, past, left_on=past.index.names, right_index=True, how="left")

	def cal_AUR(self):
		print("calculate AUR")
		self.data["AUR"] = np.where(self.data["sales_date"].isin([fcst_date, pd.to_datetime(fcst_date)]),
									self.data.eval("MSRP * md").round(),
									self.data.eval("sales_amt / sales_qty").round())

	def price_band(self):
		print("calculate price band")
		self.data["price_band"] = pd.cut(self.data["MSRP"], [0, 300, 500, 700, 1000, self.data["MSRP"].max()],
										 labels=[300, 500, 700, 1000, 3000])
		self.data["AUR_band"] = pd.cut(self.data["AUR"], [0, 200, 400, 600, 800, 1000, self.data["AUR"].max()],
									   labels=[200, 400, 600, 800, 1000, 3000])
		self.data["MSRP_pos"] = (self.data["MSRP"] - self.data["MSRP"].min()) / (
					self.data["MSRP"].max() - self.data["MSRP"].min())

	def create_features(self, shift_days=14):
		# 回测阶段shift days是7或者14， 表示要空出来7或者14天，用来做markdown plan, 即提前14天做打折计划
		# self.cal_md()								# round to 0.1
		# if self.level == "prod_code":
		# 	self.extract_color()
		self.same_md_hist_frequency()  # rm include
		self.inseason_same_md_hist_frequency()  # rm include
		self.hist_same_md_aps_stats(shift_days=shift_days)
		self.cal_AUR()
		self.price_band()
		self.cal_inseason_this_md_previous_aps(shift_days=shift_days)
		self.cal_hist_md_avg()
		# self.cal_all_md_recent_aps()
		self.cal_all_md_prior_7d_aps(shift_days)
		self.cal_md_prior_7d_aps(shift_days)
		self.cal_prior_7d_full_price_aps(shift_days)


# competing styles features
def calculate_in_group(df, grp_name="", price_column='MSRP', rank_column="AUR", md_column="md", range=100,
					   ascending=True):
	reverse_flag = True if not ascending else False
	new_features = []
	for idx, row in df.iterrows():
		px = row[price_column]
		md = row[f'weighted_{md_column}']
		filtered = df.query(f"abs({price_column} - @px) <= @range")
		new_features.append([
			md / filtered[f'weighted_{md_column}'].mean() if filtered[f'weighted_{md_column}'].mean() > 0 else 999,
			# inv都是0，不可卖
			len(filtered),
			round(sorted(filtered[rank_column], reverse=reverse_flag).index(row[rank_column]) / len(filtered), 2),
		])
	return pd.DataFrame(new_features, index=df.index,
						columns=[f"same_{grp_name}_{price_column}_{md_column}_competitiveness",
								 f"same_{grp_name}_{price_column}_comp_cnt",
								 f"same_{grp_name}_{price_column}_{rank_column}_rank"])


def calculate_competing(df, same_columns, grp_name="", price_column='MSRP', md_column="md", inventory_column="inv_qty",
						rank_column="AUR", range=100, ascending=True):
	df[f'weighted_{md_column}'] = df.eval(f"{md_column} * {inventory_column}")
	groups = df.groupby(same_columns)
	output = []
	for level, group in tqdm(groups):
		output.append(
			calculate_in_group(group, grp_name, price_column, rank_column, md_column, range=range, ascending=ascending))
	output = pd.concat(output)
	del df[f'weighted_{md_column}']
	return pd.merge(df, output, left_index=True, right_index=True, how="left")


def load_existing_competing_style_features(ftw_model_att3):
	print("load existing competing styles features")
	try:
		if fcst_level == "style":
			a = pd.read_csv(os.path.join(paths.step_data, competing_styles_existing_features))
		else:
			a = pd.read_csv(os.path.join(paths.step_data, competing_skus_existing_features))
		a["sales_date"] = pd.to_datetime(a["sales_date"])
		if a.shape[0] > 0:
			to_date = a.sales_date.max()
			start_date = a.sales_date.min()
			old = ftw_model_att3.query("(sales_date <= @to_date) & (sales_date >= @start_date)")
			if old.shape[0] != a.shape[0]:
				print(
					"different data shape within same time scope compared with existing features（product or days is not the same）")
			new = ftw_model_att3.query("(sales_date > @to_date) | (sales_date < @start_date)")
			old_finish = pd.merge(old, a, on=['sales_date', level_col, 'inseason_flag1', "season1"], how="left")
		else:
			print(f"no existing competing {fcst_level} features available")
			old_finish = None
			new = ftw_model_att3
	except:
		print(f"can't find existing competing {fcst_level} features file")
		old_finish = None
		new = ftw_model_att3
	return old_finish, new


def competing_styles_features(data, step_data_save_path):
	"""
		符合某种竞品定义的竞品数量
		同一个competing styles group里本style相对优势：same MSRP下md的优势，same AUR 下的 MSRP优势
		过去14天full price aps在竞品中的排位
	"""
	print("calculate competing features")
	ftw_model_att3 = data.copy()
	ftw_model_att3["POWER FRANCHISE/ITEM"] = ftw_model_att3["POWER FRANCHISE/ITEM"].fillna("non-power-franchise")
	ftw_model_att3.rename(columns={f"past_7_full_price_avg_aps": f"past_7_aps"}, inplace=True)
	ftw_model_att3[f"past_7_aps"] = ftw_model_att3[f"past_7_aps"].fillna(0)
	grp_index = ["sales_date", "Gender Group", 'GC Category']
	# 已经计算过competing style的record直接把之前的拿过来， 没计算过的参与计算
	old_finish, new = load_existing_competing_style_features(ftw_model_att3)
	if len(new) > 0:
		new = calculate_competing(new, grp_index)
		new = calculate_competing(new, grp_index, "", "AUR", "MSRP", "inv_qty", f"past_7_aps",
								  range=100, ascending=False)
		# same date/gender/category/platform
		grp_index_platform = grp_index + ["Classification Platform"]
		new = calculate_competing(new, grp_index_platform, "platform", "AUR", "MSRP", "inv_qty",
								  f"past_7_aps", range=100, ascending=False)
		# same date/gender/category/power franchise
		grp_index_power_franchise = grp_index + ["POWER FRANCHISE/ITEM"]
		new = calculate_competing(new, grp_index_power_franchise, "pc", "AUR", "MSRP", "inv_qty",
								  f"past_7_aps", range=100, ascending=False)
		# same date/gender/category/model
		grp_index_model = grp_index + ["Model"]
		new = calculate_competing(new, grp_index_model, "Model", "AUR", "MSRP", "inv_qty",
								  "past_7_aps", range=100, ascending=False)
		if fcst_level == "sku":
			# same date/gender/category/color
			grp_index_color = grp_index + ["base_color"]
			new = calculate_competing(new, grp_index_color, "color", "AUR", "MSRP", "inv_qty",
									  "past_7_aps", range=100, ascending=False)
		# 合并新旧数据
		ftw_model_att3 = pd.concat([old_finish, new])
		ftw_model_att3 = ftw_model_att3.sort_values(by=["sales_date", level_col])
		ftw_model_att3.to_csv(f"{step_data_save_path}/{fcst_level}_attr_data5.csv", index=False)

		# 存储competing styles的features到库里
		competing_save_cols = ['sales_date', 'season1', 'inseason_flag1', level_col]
		if fcst_level == "style":
			competing_save_cols.extend(competing_features_cols)
			ftw_model_att3[competing_save_cols].query("sales_date != @fcst_date").to_csv(
				os.path.join(paths.step_data, competing_styles_existing_features),
				index=False)
		else:
			competing_save_cols.extend(competing_features_cols_sku)
			ftw_model_att3[competing_save_cols].query("sales_date != @fcst_date").to_csv(os.path.join(paths.step_data, competing_skus_existing_features),
													   index=False)
	else:
		ftw_model_att3 = old_finish
	return ftw_model_att3


# after merge with Sophie's data
def convert_date_col_to_days_col(src_data):
	data = src_data.copy()
	data["days_from_inseason_1st_day"] = (
				pd.to_datetime(data["sales_date"]) - pd.to_datetime(data["inseason_1st_day"])).dt.days
	data["season_begin_date"] = data["season1"].replace(season_start_dict)
	data["days_from_season_begins"] = (
				pd.to_datetime(data["sales_date"]) - pd.to_datetime(data["season_begin_date"])).dt.days
	data["days_from_hist_1st_day"] = (
				pd.to_datetime(data["sales_date"]) - pd.to_datetime(data['first_sales_date'])).dt.days
	data["days_from_this_md_first_day"] = (
				pd.to_datetime(data["sales_date"]) - pd.to_datetime(data['this_md_first_day'])).dt.days
	return data


def merge_site_traffic(style_features):
	print("merge traffic")
	traffic = pd.read_excel(f"{paths.source}/{md_sensitivity_model_traffic_file}")
	tmall_traffic = get_tmall_traffic(traffic)
	return pd.merge(style_features, tmall_traffic, left_on="sales_date", right_index=True, how="left")


def merge_style_traffic(style_features):
	# style_traffic = pd.read_excel(os.path.join(paths.source, style_traffic_daily_update)) # 新的traffic数据 缺失76%
	# style_traffic = style_traffic[["TRAN_DT", "STYLCOLOR_CD", "PDP_TRAFFIC"]]
	# style_traffic.columns = ["sales_date", "style_cd", "style_traffic"]
	style_traffic = pd.read_csv(f"{paths.source}/{style_traffic_daily}")  # 旧的traffic数据
	style_traffic["sales_date"] = pd.to_datetime(style_traffic["sales_date"])
	style_traffic = style_traffic.groupby(["style_cd", "sales_date"])["style_traffic"].mean()
	return pd.merge(style_features, style_traffic, left_on=style_traffic.index.names, right_index=True, how="left")


def get_sophie_features():
	print("merge sophie's features")
	if fcst_level == "style":
		sophie_feature = pd.read_csv(f"{paths.source}/{sophie_features_src}")
	else:
		sophie_feature = pd.read_csv(f"{paths.source}/{sophie_features_sku_src}")
	sophie_feature = sophie_feature.drop(['Unnamed: 0', 'sales_qty', 'sales_amt', 'MSRP', 'inv_qty', 'Gender Group',
										  'GC_Category', 'season1', 'SEASON', 'Subcategory', 'md_price', 'md',
										  'comp_price_low', 'comp_price_high', 'group'], axis=1)
	sophie_feature.rename(columns={'Style_Number': "style_cd"}, inplace=True)
	sophie_feature["sales_date"] = pd.to_datetime(sophie_feature["sales_date"])
	return sophie_feature


def shift_inventory_for_model_training(src_data, pass_days):
	print("get inventory feature")
	inventory = pd.read_csv(os.path.join(paths.step_data, local_inventory_db))
	inventory["sales_date"] = pd.to_datetime(inventory["EOP_DT"]) + \
							  datetime.timedelta(days=1) + datetime.timedelta(days=pass_days)
	inventory.rename(columns={"STYLCOLOR_CD": "prod_code", "EOP_QTY": "inv_qty"}, inplace=True)
	inventory_sku_by_platform = inventory.groupby(["platform", "prod_code", "sales_date"])[
		"inv_qty"].sum().reset_index()
	del src_data["inv_qty"]  # inital inv_qty is one day prior sales_date col, which is fcst day
	src_data["sales_date"] = pd.to_datetime(src_data["sales_date"])
	src_data = pd.merge(src_data, inventory_sku_by_platform, on=["platform", "sales_date", "prod_code"], how="left")
	return src_data


def prepare_modeling_data(src_data, shift_days):
	"""
	:param src_data: the data master file after running md_sensitivity_model_data_init.py
	:param pass_days: if data is prepared for training model, then pass_days should be lead time, if data is prepared for forecast, then pass_days should be 0
	:return:
	"""
	ftw_src = src_data.query("Division == @PE")
	ftw_src = shift_inventory_for_model_training(ftw_src,
												 shift_days)  # sales_date对应的是要预测的那天，所以inventory是做predict那天的前一天的inventory
	# sku to style transform
	ftw_src["style_cd"] = ftw_src["prod_code"].str[:6]
	if fcst_level == "style":
		style_ftw_src = sku_to_style_attribute_transform(ftw_src)
	else:
		style_ftw_src = ftw_src
	print("calculate sku # in same style this season/date")
	sn_style_sku_num = cal_sku_num_in_style(ftw_src,
											time_scope="season")  # sku number in this style [same_style_sku_num_sn] this season
	dt_style_sku_num = cal_sku_num_in_style(ftw_src,
											time_scope="sales_date")  # sku number in this style [same_style_sku_num_dt] this date
	style_ftw_base = pd.merge(style_ftw_src, sn_style_sku_num, left_on=sn_style_sku_num.index.names,
							  right_index=True, how="left")
	style_ftw_base = pd.merge(style_ftw_base, dt_style_sku_num, left_on=dt_style_sku_num.index.names,
							  right_index=True, how="left")
	if fcst_level == "sku":
		style_ftw_base = extract_colors(style_ftw_base)
	# calculate historical&recent sales features
	feature_maker = FeatureMaker(style_ftw_base, level=fcst_level, rm_remove=False)  # 不去掉9.9等大型retail moments
	feature_maker.create_features(shift_days=shift_days)
	# feature_maker.data.to_csv(f"{getattr(paths, f'{fcst_level}_model_step_data')}/{fcst_level}_attr_data_before_competing.csv", index=False)
	## features for competing styles with different competing group definition
	style_features = competing_styles_features(feature_maker.data,
											   step_data_save_path=getattr(paths, f"{fcst_level}_model_step_data"))
	## traffic features
	style_features = merge_site_traffic(style_features)
	if fcst_date_site_traffic:
		style_features["site_traffic"] = np.where(style_features["sales_date"] == fcst_date, fcst_date_site_traffic,
												  style_features["site_traffic"])
	# style_features = merge_style_traffic(style_features)  # 严重缺失 70%
	style_features = cal_md_round(style_features, level=level_col)
	style_features = cal_hist_max_qty(style_features)  # 为了改进大促的表现
	style_features = cal_sn_to_date_st(style_features)
	style_features.to_csv(f"{getattr(paths, f'{fcst_level}_model_step_data')}/{fcst_level}_attr_data_before_sophie.csv",
						  index=False)
	## merge with Sophie's data
	sophie_feature = get_sophie_features()
	style_features = pd.merge(style_features, sophie_feature, on=[level_col, 'sales_date', 'inseason_flag1'],
							  how="left")
	style_features = convert_date_col_to_days_col(style_features)
	style_features["st_saturate_180"] = style_features.eval("180 * sn_to_date_st / days_from_season_begins / 0.65")
	style_features["st_saturate_90"] = style_features.eval("90 * sn_to_date_st / days_from_season_begins / 0.65")
	return style_features


def cal_days_from_rm(data, season_rm_dict):
	"""
	:param data:
	:param season_rm_dict: 每个season大促开始的日期
	:return: days_since_rm 每个season每一天距离大促第一天的天数
	"""
	data["is_rm"] = np.where(data["sales_date"].isin(season_rm_dict.values()), 1, 0)
	data["days_since_rm"] = np.nan
	for season, season_rm_start_date in season_rm_dict.items():
		data["days_since_rm"] = np.where(data["season1"] == season, (
					pd.to_datetime(data["sales_date"]) - pd.to_datetime(season_rm_start_date)).dt.days,
										 data["days_since_rm"])
	return data


def filter_modeling_data(data_master, train_seasons, rm_remove=True, off_season_remove=True, full_price_remove=True):
	"""
	:param data_master:
	:param train_seasons: SP/SU/FA/HO seasons that selected for training model or sales dates for predict
	:param rm_remove: False if don't forecast sales on rm_dates(defined in config.py)
	:param off_season_remove: False if don't forecast off-season sales
	:param full_price_remove: False if don't forecast full-price products
	:return: filtered training dataset
	"""
	if train_seasons:
		# modeling_data = data_master[data_master["season1"].str.startswith(train_seasons)]
		modeling_data = data_master[data_master["season1"].isin(train_seasons)]

	else:
		modeling_data = data_master.copy()
		modeling_data["season"] = modeling_data["season1"].str[:2]
	if off_season_remove:
		# 1. inseason only
		modeling_data = modeling_data.query("(inseason_flag1 == 'Y')")
	# del modeling_data_master["inseason_flag1"]
	if rm_remove:
		# 2.只做非retail moment
		modeling_data = modeling_data[(~modeling_data["sales_date"].isin(rm_dates))]

	# 不包含全价
	if full_price_remove:
		modeling_data = modeling_data[modeling_data["md"] <= 0.95]
	# modeling_data_master = modeling_data_master.query("days_from_season_begins > 14")
	# delete cols that are useless for modeling
	drop_key = {"prod_code", "style_cd"} - {level_col}
	modeling_data = modeling_data.drop(
		["inseason_1st_day", "season_begin_date", 'first_sales_date', 'this_md_first_day',
		 "platform", 'Full Color Description', "sales_amt", 'SEASON', "Division", 'Style Number', 'CN Retail Price',
		 'First Offer Date', 'Classification Detail', 'season1', drop_key.pop()], axis=1)
	return modeling_data


class ModelData:

	def __init__(self, data, level="prod_code", is_forecast=False):
		self.data_master = data.copy()
		# self.model_data = data[data["days_from_season_begins"] >= days_begin]      # days_from_prod_inseason_1st_day
		self.model_data = data.copy()
		self.level = level
		self.is_forecast = is_forecast

	def get_model_data(self, y_numer, y_denom=None):
		if y_denom:
			self.y_col = f"{y_numer}_over_{y_denom}"
			self.model_data[self.y_col] = self.model_data.eval(f"{y_numer}/{y_denom}")
		else:
			self.y_col = y_numer  # 直接预测销量

		self.numerical_cols = set(self.model_data.describe().columns) - set("Material Intent")
		self.categorical_cols = set(self.model_data.columns) - set(self.numerical_cols) - {"sales_date",
																						   'Marketing Name', self.level}
		for col in self.categorical_cols:
			self.model_data[col] = pd.Categorical(self.model_data[col])

		if not self.is_forecast:
			self.model_data = self.model_data[
				(self.model_data[f"{self.y_col}"].notnull()) & (self.model_data[f"{self.y_col}"] != np.inf)]
			self.y_data = np.log(self.model_data[f"{self.y_col}"] + 1)
		else:
			self.y_data = self.model_data[y_denom]
			self.model_folder = getattr(paths, f'model_{self.y_col}')
		self.x_data = self.model_data[list(set(self.model_data.columns) - {"sales_date", self.level, 'Marketing Name',
																		   y_numer, self.y_col})]
		self.x_data["AUR_band"] = self.x_data["AUR_band"].astype(int)
		self.x_data["price_band"] = self.x_data["price_band"].astype(int)

	def check_features(self, folder):
		global paths
		paths.add_source(f"model_{self.y_col}", os.path.join(folder, self.y_col))
		self.model_folder = getattr(paths, f'model_{self.y_col}')
		# data check
		writer = pd.ExcelWriter(f"{getattr(paths, f'model_{self.y_col}')}/features_check.xlsx")
		self.x_data.describe().T.to_excel(writer, "describe")
		self.na_rate.to_excel(writer, "nan_check")
		pd.Series(self.x_data.columns).to_excel(writer, "features")
		# y_cut = pd.qcut(self.y_data,10)
		# y_cut.value_counts().sort_values(ascending=False).to_excel(writer,"y_decile")
		writer.close()
		self.model_data.to_excel(f"{getattr(paths, f'model_{self.y_col}')}/model_data.xlsx", index=False)

	def remove_almost_nan_features(self, threshold=0.95):
		self.na_rate = self.x_data.isnull().sum() / len(self.x_data)
		na_cols = self.na_rate[self.na_rate >= threshold].index
		self.x_data = self.x_data[[x for x in self.x_data.columns if x not in na_cols]]
		print(f"x_data size: {self.x_data.shape}")


class MDModel:
	def __init__(self, y_denom, data_manager, train_ratio, is_forecast=False):
		self.x_data = data_manager.x_data
		self.y_data = data_manager.y_data if not is_forecast else None
		self.inv_adj = False
		self.perf_dict = {}
		self.y_denom = y_denom
		self.data_manager = data_manager
		self.train_test_ratio = train_ratio
		self.is_forecast = is_forecast

	def train_test_split(self):
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.x_data, self.y_data,
																				train_size=self.train_test_ratio,
																				random_state=82)

	def run_lgb(self, **params):
		lgb_train = lgb.Dataset(self.X_train, label=self.y_train,
								categorical_feature=[x for x in self.data_manager.categorical_cols if
													 x in self.x_data.columns])
		lgb_test = lgb.Dataset(self.X_test, label=self.y_test,
							   categorical_feature=[x for x in self.data_manager.categorical_cols if
													x in self.x_data.columns])
		lgb_params = {
			'task': 'train',
			'boosting_type': 'gbdt',
			'objective': 'regression',
			# 'is_unbalance': 'true',
			'metric': 'l1',
			'learning_rate': params["lr"],
			"n_estimators": params["n"],
			'max_depth': params["depth"],
			"num_leaves": params["num_leaves"]
		}
		self.clf = lgb.train(lgb_params, lgb_train, valid_sets=[lgb_test])
		self.feature_importance = pd.DataFrame(zip(self.clf.feature_name(), self.clf.feature_importance()),
											   columns=["features", "importance"]).sort_values(by="importance",
																							   ascending=False)

	def merge_predict_result_data(self):
		self.predict_result = pd.concat([self.data_manager.model_data,
										 pd.Series(np.exp(self.clf.predict(self.X_test)) - 1, name="predict",
												   index=self.X_test.index)], axis=1)
		self.predict_result["predict"] = self.predict_result["predict"].apply(
			lambda x: 0 if x < 0 else x)  # sales boost 不应该小于0
		if self.y_denom:
			self.predict_result["predict_sales_qty"] = self.predict_result.eval(f"{self.y_denom}*predict").round()
		else:
			self.predict_result.rename(columns={"predict": "predict_sales_qty"}, inplace=True)
		if not self.is_forecast:
			self.predict_result["MAPE_qty"] = abs(
				self.predict_result["predict_sales_qty"] - self.predict_result["sales_qty"]) / self.predict_result[
												  "sales_qty"]
			self.predict_result["is_test"] = np.where(self.predict_result["predict_sales_qty"].notnull(), 1, 0)
			self.predict_result["MAPE*qty"] = self.predict_result.eval("sales_qty*MAPE_qty")
			self.predict_result["mape_below_0.3"] = np.where(self.predict_result["MAPE_qty"] <= 0.3, 1, 0)
			self.predict_result["predict-real"] = (self.predict_result["predict_sales_qty"] - self.predict_result[
				"sales_qty"]) / self.predict_result["sales_qty"]

	def test_sales_qty_mape(self):
		test_mape = (abs(self.predict_result["predict_sales_qty"] - self.predict_result["sales_qty"]) /
					 self.predict_result["sales_qty"]).median()
		print(f"sales_qty test mape: {test_mape}")
		self.perf_dict["sales qty test mape"] = test_mape
		test_size = self.predict_result.query("is_test > 0")
		self.perf_dict["销量占比 MAPE<=30%"] = test_size[test_size["mape_below_0.3"] == 1]["sales_qty"].sum() / test_size[
			"sales_qty"].sum()
		self.perf_dict["鞋次占比 MAPE<=30%"] = test_size[test_size["mape_below_0.3"] == 1]["sales_date"].count() / \
										   test_size["sales_date"].count()
		print(self.perf_dict)

	def test_sales_boost_performance(self):
		if not self.y_denom:
			raise ValueError("this model predict sales qty directly, no need for sales boost performance!")
		self.predict_result["real_boost"] = pd.qcut(self.predict_result[self.data_manager.y_col].round(), 15,
													duplicates="drop")
		c = self.predict_result["real_boost"].dtype.categories
		c = sorted(set(chain(c.left, c.right)))
		self.predict_result["predict_boost"] = pd.cut(self.predict_result["predict"], c)
		self.matrix = pd.crosstab(self.predict_result["real_boost"], self.predict_result["predict_boost"])
		print(f"salesBoost performance: {np.diag(self.matrix).sum() / self.matrix.sum().sum()}")
		self.perf_dict["salesBoost performance"] = np.diag(self.matrix).sum() / self.matrix.sum().sum()

	def save_result(self, name=""):
		joblib.dump(self.clf, f"{self.data_manager.model_folder}/md_sensitivity_{self.train_test_ratio}.pkl")
		pd.Series(self.x_data.columns).to_excel(
			f"{self.data_manager.model_folder}/md_sensitivity_{self.train_test_ratio}_features.xlsx")
		result_file = pd.ExcelWriter(
			f"{self.data_manager.model_folder}/predict_result_{self.train_test_ratio}{name}.xlsx")
		self.predict_result.to_excel(result_file, "predict_src")
		self.feature_importance.to_excel(result_file, "feature_importance")
		if self.y_denom:
			self.matrix.to_excel(result_file, "boost_evaluate")
		result_file.close()

	def load_model(self, md=None):
		if md:
			self.clf = md.clf
			self.use_features = md.x_data.columns
		else:
			self.clf = joblib.load(f"{self.data_manager.model_folder}/md_sensitivity_{self.train_test_ratio}.pkl")
			self.use_features = \
			pd.read_excel(f"{self.data_manager.model_folder}/md_sensitivity_{train_test_ratio}_features.xlsx")[
				0].tolist()
		self.X_test = self.x_data[self.use_features]


def run_model(modeling_data, level_col, y_numer, y_denom, params_dict, train_ratio=0.75, save_result=True):
	"""
	:param modeling_data: contains y and predictors
	:param level_col: style_cd or prod_code
	:param y_numer: sales qty col need to forecast
	:param y_denom: denominator of sales boost, this model predicts sales boost first, then transfer sales boost to sales qty to evaluate result
	:param params_dict: model params
	:param train_ratio: if < 1 if need test model performance using historical data, =1 if use all data to train model
	:param save_result: False if this is used for searching better model params; True if this is the final model
	:return: model evaluation result; data results are saved to folder
	"""
	""" for modeling params search"""
	data_manager = ModelData(modeling_data, level_col)
	data_manager.get_model_data(y_numer, y_denom)
	data_manager.remove_almost_nan_features(threshold=0.98)
	if save_result:
		data_manager.check_features(paths.model_result)

	md_model = MDModel(y_denom, data_manager, train_ratio)
	md_model.train_test_split()
	md_model.run_lgb(**params_dict)
	if train_ratio < 1:
		md_model.merge_predict_result_data()
		md_model.test_sales_qty_mape()
		if y_denom:
			md_model.test_sales_boost_performance()
		if save_result:
			md_model.save_result()
			with open(f"{data_manager.model_folder}/model_params.json", 'w') as f:
				json.dump(params_dict, f)
		performance = md_model.perf_dict
	else:
		performance = None
		print("train test split = 1, no performance available")
	return performance, md_model


def merge_zhaoxu_sku_features(data):
	"""
		for sku model only, features are related to skus' performance within same style
	"""
	data["sales_date"] = pd.to_datetime(data["sales_date"])
	zhaoxu_sku = pd.read_csv(zhaoxu_sku_features)
	zhaoxu_sku["sales_date"] = pd.to_datetime(zhaoxu_sku["sales_date"])
	return pd.merge(data, zhaoxu_sku, on=["sales_date", "prod_code", "inseason_flag1"], how="left")


def merge_last_rm_features(data):
	"""
        features related to last retail moment sales, only used for forecasting retail moment sales,
        features are non for non-retail moment days
    """
	data["sales_date"] = pd.to_datetime(data["sales_date"])
	if fcst_level == "sku":
		last_rm = pd.read_csv(os.path.join(paths.source, zx_rm_sku_features))
	else:
		last_rm = pd.read_csv(os.path.join(paths.source, zx_rm_style_features))
	last_rm["sales_date"] = pd.to_datetime(last_rm["sales_date"])
	repeat_cols = set([x for x in last_rm.columns if (x in last_rm.columns and x in data.columns)]) - {level_col,
																									   "sales_date"}
	result = pd.merge(data.drop(repeat_cols, axis=1), last_rm, on=["sales_date", level_col], how="left")
	if result.shape[0] != data.shape[0]:
		raise ValueError("data length is changed after left merge, duplicate rows ERROR for right data!")
	return result


def get_oot_performance(result, predict_qty_col='predict_sales_qty'):
	result = result[result[predict_qty_col].notnull()]
	result["fcst_amt"] = round(result[predict_qty_col]) * result["AUR"]
	result["real_amt"] = result["sales_qty"] * result["AUR"]
	result["MAPE"] = abs(round(result[predict_qty_col]) / result["sales_qty"] - 1)
	result["MAPE_below_0.3"] = result["MAPE"] <= 0.33
	print("MAPE below 0.3 by gender agg:")
	grp = result.groupby(["Gender Group"])[["fcst_amt", "real_amt", predict_qty_col, "sales_qty"]].sum()
	grp["mape_amt"] = grp.eval("fcst_amt / real_amt-1")
	grp["mape_qty"] = grp.eval("predict_sales_qty / sales_qty-1")

	print(grp)
	print("MAPE below 0.3:")
	grp2 = result.groupby("MAPE_below_0.3").agg({"real_amt": sum,
												 "sales_qty": sum,
												 level_col: 'count'})
	prct_low_mape = grp2.loc[True, :] / grp2.sum(axis=0)
	prct_low_mape.name = "MAPE below 0.3 prct"
	print(prct_low_mape)
	grp2 = pd.concat([grp2, pd.DataFrame(prct_low_mape).T], axis=0)

	result = result.sort_values(by='sales_qty', ascending=False)
	result["sales_prct"] = (result["sales_qty"] / result["sales_qty"].sum()).cumsum()
	print("median MAPE of top 30% sales qty style:")
	print(result.query("sales_prct <= 0.3")["MAPE"].median())

	grp3 = result.groupby(["Gender Group", "GC Category"])[
		["fcst_amt", "real_amt", predict_qty_col, "sales_qty"]].sum().sort_values(by="sales_qty", ascending=False)
	grp3["mape_amt"] = grp3.eval("fcst_amt/real_amt - 1")
	grp3["mape_qty"] = grp3.eval("predict_sales_qty/sales_qty - 1")

	a = result.groupby(level_col)[["sales_qty", predict_qty_col]].sum().sort_values(by="sales_qty", ascending=False)
	a["MAPE"] = abs(a[predict_qty_col] / a["sales_qty"] - 1)
	a["cum_prct"] = (a["sales_qty"] / a["sales_qty"].sum()).cumsum()
	a["median_mape"] = a["MAPE"].expanding().median()
	print("Median MAPE of top 30% styles:")
	print(a.query("cum_prct<=0.3")["MAPE"].median())
	return grp, grp2, a, grp3


def cal_sn_to_date_st(style_features):
	"""
		calculate cumulative season to date sold qty，along with its rank within same PE and GENDER
		calculate inv qty rank in same PE/GENDER
		calculate broad inv qty(include in-transit
		calculate season to date sell thru
	"""
	style_features["sales_date"] = pd.to_datetime(style_features["sales_date"])
	style_features["inv_qty_rank"] = style_features.groupby(["sales_date", "Gender Group"])["inv_qty"].rank(
		ascending=False, pct=True).round(2)
	inventory = pd.read_csv(os.path.join(paths.step_data, local_inventory_db))
	inventory = inventory.query("platform == @use_platform")
	inventory["sales_date"] = pd.to_datetime(inventory["EOP_DT"]) + datetime.timedelta(days=pass_days)
	inventory.rename(columns={"STYLCOLOR_CD": "prod_code", "EOP_QTY": "inv_qty"}, inplace=True)
	if fcst_level == 'style':
		inventory["style_cd"] = inventory["prod_code"].str[:6]
	inventory_sku_by_platform = inventory.groupby(["platform", level_col, "sales_date"])[
		"inv_qty", "IN_TRANSIT_QTY", "bz_wh_transit_qty"].sum().reset_index()
	inventory_sku_by_platform["broad_inv_qty"] = inventory_sku_by_platform.eval(
		"inv_qty + IN_TRANSIT_QTY + bz_wh_transit_qty")
	# season to date sell thru
	cum_qty = pd.DataFrame()
	for season in style_features["season1"].unique():
		for sex in style_features["Gender Group"].unique():
			season_data = style_features[
				(style_features["season1"] == season) & (style_features["Gender Group"] == sex)]
			pivot = pd.crosstab(index=season_data["sales_date"], columns=season_data[level_col],
								values=season_data["sales_qty"], aggfunc='sum').cumsum().shift(pass_days, freq='D')
			sn_cum_qty = pivot.ffill().stack()
			sn_cum_qty = pd.concat(
				[sn_cum_qty, sn_cum_qty.groupby(["sales_date"]).rank(ascending=False, pct=True).round(2)], axis=1)
			sn_cum_qty.columns = ["sn_cum_qty", "sn_cum_qty_rank"]
			cum_qty = pd.concat([cum_qty, sn_cum_qty.reset_index()])
	cum_qty = pd.merge(cum_qty, inventory_sku_by_platform[[level_col, "sales_date", "broad_inv_qty"]], how="left",
					   on=[level_col, "sales_date"])
	cum_qty["sn_to_date_st"] = cum_qty.eval("sn_cum_qty / (broad_inv_qty + sn_cum_qty)").round(2)
	return pd.merge(style_features, cum_qty, how="left", on=["sales_date", level_col])


def cal_hist_max_qty(style_features):
	# caluculate historical max qty
	pivot2 = pd.crosstab(index=style_features["sales_date"], columns=style_features[level_col],
						 values=style_features["sales_qty"],
						 aggfunc='sum').cummax().shift(pass_days, freq='D')
	cummax_qty = pivot2.ffill().stack()
	cummax_qty.name = "hist_max_qty"
	return pd.merge(style_features, cummax_qty, on=cummax_qty.index.names, how="left")


def evaluate_performance(result, save=True):
	"""
	:param result: predict result
	:param save: if save result to excel
	:return: get performance metrics of forecast
	"""
	# 预测版：保存预测数据
	if is_real_forecast:
		def get_metrics(result):
			eval_dict = {"tot qty": result["predict_sales_qty"].sum(),
						 "tot rev": result.eval("predict_sales_qty * AUR").sum(),
						 "full price #": len(result.query("md > 0.8")) / len(result),
						 "full price % qty": result.query("md > 0.8")["predict_sales_qty"].sum() / result[
							 "predict_sales_qty"].sum(),
						 "full price % amt": result.query("md > 0.8").eval(
							 "predict_sales_qty * AUR").sum() / result.eval(
							 "predict_sales_qty * AUR").sum()}
			return eval_dict
		eval_dict = get_metrics(result)
		print(eval_dict)
		gender_metrics = []
		for gender in result["Gender Group"].unique():
			gender_dt = result[result["Gender Group"] == gender]
			metrics = get_metrics(gender_dt)
			metrics["Gender"] = gender
			gender_metrics.append(metrics)
		print(gender_metrics)

		if save:
			save_file = pd.ExcelWriter(f"{paths.fcst_result}/fcst_result_{fcst_date}.xlsx")
			result.to_excel(save_file, sheet_name="src")
			pd.Series(eval_dict).to_excel(save_file, sheet_name="metrics")
			pd.DataFrame(gender_metrics).to_excel(save_file, sheet_name="metrics", startrow=9)
			save_file.save()
	# 回测版
	else:
		grp, grp2, a, grp3 = get_oot_performance(result, predict_qty_col='predict_sales_qty')
		if save:
			# save  oot result
			save_file = pd.ExcelWriter(f"{paths.fcst_result}/fcst_result_{fcst_date}_oot.xlsx")
			grp.to_excel(save_file, sheet_name="Gender agg")
			grp2.to_excel(save_file, sheet_name="mape below 0.3 prct")
			a.to_excel(save_file, sheet_name=f'top_30%_{fcst_level}')
			grp3.to_excel(save_file, sheet_name="GenderXCategory agg")
			result.to_excel(save_file, sheet_name='src')
			save_file.save()


# from imblearn.under_sampling import RandomUnderSampler
def resample_data(modeling_data, ignore_date_since, under_sample_scale=0.8, upper_sample_scale=0.2):
	data1 = modeling_data[(modeling_data["sales_date"].isin(rm_dates))
						  | (modeling_data["sales_date"] >= ignore_date_since)]
	data2 = modeling_data[~((modeling_data["sales_date"].isin(rm_dates))
							| (modeling_data["sales_date"] >= ignore_date_since))]
	data3 = data2.sample(frac=under_sample_scale)
	data4 = modeling_data[modeling_data["sales_date"].isin(rm_dates)].sample(frac=upper_sample_scale)
	modeling_data2 = pd.concat([data1, data3, data4]).reset_index(drop=True)
	return modeling_data2


if __name__ == '__main__':

	## initial set up
	fcst_date = "2020-09-09"
	fcst_date_site_traffic = 5340324    # None if don't need to make forecast traffic for fcst_date

	fcst_level = "sku"  # forecast level: style/sku
	level_col = "style_cd" if fcst_level == "style" else "prod_code"
	PE = 'FOOTWEAR DIVISION'

	additional_name = "_for99_B"
	paths.add_source(f"{fcst_level}_model", f"md_sensitivity_model/{fcst_level}_model{additional_name}")
	paths.add_source(f"{fcst_level}_model_step_data", f"md_sensitivity_model/{fcst_level}_model{additional_name}/step_data")

	###### step1:  feature engineering(part 3/3)  ######
	# src_data = pd.read_csv(f"{paths.step_data}/tmall_sku_daily_data_master_fcst.csv") # run md_sensitivity_model_data_init.py firsr
	src_data = pd.read_csv(f"{paths.step_data}/{data_master_include_forecast_day_B}") # 包含了要预测当天的数据，sales_qty是空的
	style_features = prepare_modeling_data(src_data, pass_days) #每行的sales_qty是要预测的那天的y,涉及到过去x天的销量的变量，都需要空出一个准备的时间
	style_features = cal_days_from_rm(style_features, season_rm_dict) # 如果这个season没有大促，可以ignore这两个feature
	style_features = merge_last_rm_features(style_features) # 如果要预测大促，需要加上历史大促的销量(sku model加上有显著效果，style模型可不加，效果一般）
	style_features.to_csv(f"{getattr(paths, f'{fcst_level}_model_step_data')}/{md_sensitivity_model_master_file}",
						  index=False)

	###### step2:  training model ######
	"""
		# only for training model, not applicable for forecasting
		This model forecast every-day sales qty in selected seasons
		Predictors covered all-season data
	"""
	# load data
	style_features = pd.read_csv(f"{getattr(paths, f'{fcst_level}_model_step_data')}/{md_sensitivity_model_master_file}")  # step1的结果

	is_real_forecast = True   # True if predict a real future date, False if test a date exist in data(style_features)
	filter_modeling_data_dict = {'train_seasons': ["FA2019", "FA2018", "SU2020", "FA2020"],
								 'rm_remove': False,
								 'off_season_remove': False,
								 'full_price_remove': False
								 }
	modeling_data = filter_modeling_data(style_features, **filter_modeling_data_dict) # if contains rm, off-season, full-price product

	fcst_date = "2020-09-09"
	exclude_date_since = pd.to_datetime(fcst_date) - datetime.timedelta(days=pass_days)	# (如果sales_qty是nan，建模时会被自动剔除, 这里的意思是让exclude_data_since后的数据不参与建模)

	modeling_data["sales_qty"] = np.where(pd.to_datetime(modeling_data["sales_date"]) > exclude_date_since, np.nan, modeling_data["sales_qty"])
	modeling_data2 = resample_data(modeling_data,  ignore_date_since='2020-04-01', under_sample_scale=0.75, upper_sample_scale=0.15)

	# train model
	paths.add_source("model_result", os.path.join(getattr(paths,f"{fcst_level}_model"), "model_result"))
	y_numer = "sales_qty"
	y_denom = "past_3_median_aps"		# y for this model is sales boost(y_numer/y_denom), set to None if forecast sales qty itself
	model_params = {"lr": 0.07, "n": 900, "depth": 13, "num_leaves": 45}
	train_test_ratio = 0.99
	perf, model = run_model(modeling_data2, level_col, y_numer, y_denom, params_dict=model_params,
							train_ratio=train_test_ratio, save_result=True)
	paths.save_paths(getattr(paths, f"{fcst_level}_model"))


	######   Step3. Make Future Prediction   ######
	"""
		current version only support forecast one day
		and this day - date of fcst features should = pass_days(because model is training in this way)
		if forecast day is 21 days away from date of making prediction, then pass_days should set to 21 
		and rerun feature engineering processe and also retrain data
	"""
	paths.load_paths(getattr(paths, f"{fcst_level}_model"))     # load file saving location
	# 取出要预测的数据
	# style_features = pd.read_csv(f"{getattr(paths, f'{fcst_level}_model_step_data')}/{md_sensitivity_model_master_file}"
	fcst_data = style_features.query("sales_date == @fcst_date")
	predict_data = filter_modeling_data(fcst_data, **filter_modeling_data_dict)
	data_manager2 = ModelData(predict_data, level_col, is_forecast=True)
	data_manager2.get_model_data(y_numer, y_denom)
	md_model2 = MDModel(y_denom, data_manager2, train_test_ratio, is_forecast=True)
	# md_model2.load_model()             # 从本地load已经训练好的模型
	md_model2.load_model(md=model)       # 直接用上一步训练的
	md_model2.merge_predict_result_data()
	result = md_model2.predict_result
	paths.add_source("fcst_result", os.path.join(data_manager2.model_folder, f"fcst_result"))
	evaluate_performance(result, save=True)


