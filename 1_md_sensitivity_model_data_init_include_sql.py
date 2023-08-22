import glob
import os
import pandas as pd
import datetime
from config import *
import pymssql

paths.add_source("increment", os.path.join(paths.source, "increment"))

def get_increment_data(data_to_date, data_end_date):
	## read and save incremental data from foundation
	conn = pymssql.connect(host='R4WSQLP-BI',database='DGT')
	sql_trsc = """select dimension, territory, platform, season1, inseason_flag1, prod_code, size_cd, sales_date, 
					reg_msrp as MSRP, sum(sales_qty) as sales_qty, sum(sales_amt) as sales_amt
					from DGT.DBO.V_DIG_SALES_TRANS 
					where platform <> 'HK.COM' and sales_date > '@date_a' and sales_date <= '@date_b'
					group by dimension, territory, platform, season1, inseason_flag1, prod_code, size_cd, sales_date, reg_msrp"""
	sql_inv = """select * from DBO.BZ_DIGITAL_INVENTORY
	            where EOP_DT > '@date_a' and EOP_DT <= '@date_b' 
	            and store_nm in ('GROUP PURCHASE', 'L&F S-SDC DIGITAL', 'Nike.com', 
	                             'NIKE.COM-NORTH', 'Tmall','TMALL Jordan','TMALL YA')""" #sql server的 between and包含边界值
	sql_traffic = """select * from DBO.Digital_Traffic_Count"""
	 
	# transaction
	trsc = pd.read_sql(sql_trsc.replace('@date_a', data_to_date).replace('@date_b', data_end_date), conn)
	# inventory
	inv_date_start = str(pd.to_datetime(data_to_date) - datetime.timedelta(days=1) - datetime.timedelta(days=pass_days))
	inv = pd.read_sql(sql_inv.replace('@date_a', inv_date_start).replace('@date_b', data_end_date), conn)
	# traffic
	traffic = pd.read_sql(sql_traffic, conn)

	trsc.to_csv(f"{paths.increment}/V_DIG_SALES_TRANS_data.csv",index=False)
	inv.to_csv(f"{paths.increment}/inventory_tmall_nikecom_sp18_sp20.csv",index=False)
	traffic.to_excel(f"{paths.source}/{md_sensitivity_model_traffic_file}",index=False)
	conn.close()


def process_attribute_raw():
	# attribute data(MPM data from Will Sun)
	file_list = glob.glob(os.path.join(paths.source, "new_data_src20200428/mpm_attribute_from_will/*.xlsx"))
	mpm_attributes = pd.read_excel(os.path.join(paths.source,"new_data_src20200428/mpm_attr_using_cols_sample.xlsx"))
	use_cols = mpm_attributes.columns

	attribute_data = pd.DataFrame()
	for file in file_list:
	    data = pd.read_excel(file)
	    attribute_data = pd.concat([attribute_data, data[use_cols]])

	attribute_data["season_rank"] = attribute_data.apply(lambda x: x["SEASON"][:2]+"20"+x["SEASON"][-2:],axis=1).replace(dict(zip(season_list,range(len(season_list)))))
	attribute_data = attribute_data.sort_values(by=["season_rank"])
	unique_sku_attributes = attribute_data.drop_duplicates(subset=["Product Code"], keep="last")
	unique_sku_attributes.rename(columns={"Product Code": "prod_code"}, inplace=True)
	return unique_sku_attributes


def merge_raw_data():
	# ______ read increment data
	# TMALL&NIKE.com daily sold transaction data
	daily_sales = pd.read_csv(os.path.join(paths.increment, "V_DIG_SALES_TRANS_data.csv"))
	tmall_sold = daily_sales.query("platform == @use_platform")
	tmall_sku_daily_sales = tmall_sold.groupby(["platform", "sales_date", "prod_code", "season1", "inseason_flag1"]).agg({"sales_qty":sum,"sales_amt":sum, "MSRP":pd.Series.median})
	tmall_sku_daily_sales = tmall_sku_daily_sales.reset_index()
	tmall_sku_daily_sales["sales_date"] = pd.to_datetime(tmall_sku_daily_sales["sales_date"])

	unique_sku_attributes = process_attribute_raw()
	tmall_sku_daily_sales_attr = pd.merge(tmall_sku_daily_sales, unique_sku_attributes, on="prod_code", how="left")
	tmall_sku_daily_sales_attr["md"] = tmall_sku_daily_sales_attr.eval("sales_amt/sales_qty/MSRP").round(2)
	tmall_sku_daily_sales_attr["md"] = np.where(tmall_sku_daily_sales_attr["md"] > 1, 1, tmall_sku_daily_sales_attr["md"])

	# inventory
	inventory = pd.read_csv(os.path.join(paths.increment, "inventory_tmall_nikecom_sp18_sp20.csv"))
	inventory["platform"] = inventory["STORE_NM"].replace(platform_dict)
	inventory["sales_date"] = pd.to_datetime(inventory["EOP_DT"]) + datetime.timedelta(days=1) # 算sales前一天的inventory，跟实际的transaction sales date merge
	inventory.rename(columns={"STYLCOLOR_CD":"prod_code", "EOP_QTY":"inv_qty"}, inplace=True)
	inventory_sku_by_platform = inventory.groupby(["platform","prod_code","sales_date"])["inv_qty"].sum().reset_index()

	# ____merge increment data and stock data and save to csv
	tmall_sku_daily_data_increment = pd.merge(tmall_sku_daily_sales_attr, inventory_sku_by_platform, on=["platform","sales_date","prod_code"],how="left")
	return tmall_sku_daily_data_increment


def update_local_inventory_file():
	inventory_old = pd.read_csv(os.path.join(paths.step_data, "inventory_data.csv"))		# 存量inventory
	inventory_new = pd.read_csv(os.path.join(paths.increment, "inventory_tmall_nikecom_sp18_sp20.csv")) # 增量inventory
	inventory_new["platform"] = inventory_new["STORE_NM"].replace(platform_dict)
	inventory_merge = pd.concat([inventory_old, inventory_new])
	inventory_merge["EOP_DT"] = pd.to_datetime(inventory_merge["EOP_DT"])
	inventory_merge = inventory_merge.drop_duplicates()
	inventory_merge.to_csv(os.path.join(paths.step_data, f"{local_inventory_db}"))   # 新的存量inventory


def get_retail_moment_data():
    conn = pymssql.connect(host='R4WSQLP-BI',database='DGT')
    rm = "','".join(rm_dates)
    sql_rm = f"select dimension, territory, platform, season1, inseason_flag1, " \
             f"prod_code, sales_date, reg_msrp as MSRP, sum(sales_qty) as sales_qty, " \
             f"sum(sales_amt) as sales_amt "  \
             f"from DGT.DBO.V_DIG_SALES_TRANS " \
             f"where platform <> 'HK.COM' and sales_date in ('{rm}') " \
             f"group by dimension, territory, platform, season1, inseason_flag1, " \
             f"prod_code, sales_date, reg_msrp"
    rm_data = pd.read_sql(sql_rm, conn)
    rm_data.to_csv(os.path.join(paths.source, f"{local_retail_moment_only_data_src}"), index=False)
    conn.close()


if __name__ == '__main__':

	# data master preparation for training model
	use_platform = 'TMALL'
	tmall_sku_daily_master_stock = pd.read_csv(f"{paths.step_data}/{ruofei_feature_src}") # 存量的data master:transaction + prod attribute + inventory
	data_to_date = max(tmall_sku_daily_master_stock['sales_date']) #存量截至date
	data_end_date = '2020-12-31'       # 确定取数的截止日期
	get_increment_data(data_to_date, data_end_date)   # 获取增量数据
	update_local_inventory_file()					  # 更新本地存量的库存数据
	get_retail_moment_data()						  # 计算历史大促表现相关的features用
	tmall_sku_daily_data_increment = merge_raw_data() # 对增量的原始数据做合并
	columns = tmall_sku_daily_master_stock.columns
	tmall_sku_daily_data = pd.concat([tmall_sku_daily_master_stock[columns], tmall_sku_daily_data_increment[columns]])    # 新的存量data master
	tmall_sku_daily_data.to_csv(f"{paths.step_data}/{ruofei_feature_src}",index=False)  # fcst_data

	# data master preparation for forecasting model
	# need to append forecast-day data(product, md, sales_date are input)
	# 拿到要预测的产品打折数据后，假设预测数据是2020-07-20
	fcst_data = pd.read_csv(os.path.join(paths.step_data, "plan_99_products.csv"))
	fcst_data["sales_qty"], fcst_data["sales_amt"], fcst_data["inv_qty"] = None, None, None # 要预测的当天没有sales_qty, sales_amt, 和前一天的qty
	unique_sku_attributes = process_attribute_raw()
	fcst_data_attr = pd.merge(fcst_data, unique_sku_attributes, on="prod_code", how="left")
	fcst_data_attr["sales_date"] = pd.to_datetime("2020-07-20")
	fcst_prod_list = fcst_data_attr["prod_code"].tolist()

	fcst_base_data = pd.read_csv(f"{paths.step_data}/{ruofei_feature_src}")
	# fcst_base_data = fcst_base_data[fcst_base_data["prod_code"].isin(fcst_prod_list)]
	fcst_final_data = pd.concat([fcst_base_data, fcst_data_attr[fcst_base_data.columns]])
	fcst_final_data["sales_date"] = pd.to_datetime(fcst_final_data["sales_date"]).astype(str)
	fcst_final_data.to_csv(f"{paths.step_data}/{data_master_include_forecast_day}", index=False)


