#@Time    :2/28/2020
#@Author  : Ruofei

import pandas as pd
import numpy as np
import inspect
import json
import os

pass_days = 32  # num of days ahead of promotion time (lead time for md planning),if is_forecast=True, pass_days would be 0 even if this parameter is set to a non zero value
use_platform = 'TMALL'

# 属于retail moment的sales_dates
rm_dates = ['2017-03-06','2017-03-07','2017-03-08','2017-06-18','2017-06-19','2017-06-20','2017-09-09',
	'2017-09-10','2017-11-11','2017-12-12','2018-03-07','2018-03-08','2018-03-09','2018-06-01',
	'2018-06-16','2018-06-17','2018-06-18','2018-09-09','2018-09-10','2018-11-11','2018-12-12',
	'2019-03-07','2019-03-08','2019-03-09','2019-06-01','2019-06-02','2019-06-16','2019-06-17',
	'2019-06-18','2019-09-09','2019-09-10','2019-11-11','2019-12-12','2020-06-01','2020-06-02',
	'2020-06-03','2020-06-16','2020-06-17','2020-06-18','2020-09-09','2020-09-10','2020-08-06']

# rm information setup (如果season里有大促，可以提供season里大促的开始日期,假如是否大促，和距离大促天数两个变量)
season_rm_dict = {"FA2018": "2018-09-09", "FA2019": "2019-09-09", "FA2020": "2020-09-09"}

season_list = []
for year in [2018, 2019, 2020, 2021, 2022, 2023]:
    for season in ["SP", "SU", "FA", "HO"]:
        season_list.append(f"{season}{year}")

calendar_file = "calendar_date_weekno.xlsx"
data_master_file = "data_master2020310.csv"     # sku level sales qty & attributes & site traffic & booking, this is the basic data for modeling
daily_sold_data = 'prophet_source_data.xlsx'    # daily sold data for common sku in su2018 and su2019
# source
retail_moment_file_src = "20200302_Dig_RetailMoment_Days_SP18_HO19_v1.0.csv"        # Elaine根据traffic整理出来的retail moment days
rm_days_advanced_file_src = "20200302_Dig_RetailMoment_Days_SP18_HO19_ComImported_MoreDaysAdded_v2.0.csv"       # 基于上一版和nd的promotion calendar修改，加上了更多小节日，可能并没有反映在traffic上
rm_daily_file_src = "20200316_Dig_Daily_RetailMoment_Calendar_SP18_HO19_v1.0.xlsx"

traffic_platform = "20200228_Traffic_by_SubPlatform_Week_v1.0.csv"
traffic_style = "20200305_Tmall_StyleLevel_Traffic_by_Week_v1.0.csv"
style_traffic_daily = "style_traffic_20180326_to_20190930.csv"  # FA缺失30%
style_traffic_daily_update = "style_traffic/Tmall traffic data - 2019.7.1~2019.12.31.xlsx"

feature_guide = "feature_engineering/20200526_FeatureEngineering_master.xlsx"

# rm calendar compiled by 朝旭
rm_calendar_src = "calendar/rm_flag.xlsx"
# grace comp list
comp_model_file = "IS FCST MAPPING_grace.xlsx"

attribute_use_cols = ['STYLCOLOR_CD', 'STYLCOLOR_DESC', 'PE', 'GBL_CAT_SUM_DESC', 'RETL_GNDR_GRP', 'Classification','CHN_LATEST_MSRP']
booking_use_cols = ['PLATFORM_DIMENSION', 'SEASON_DESC','STYLCOLOR_CD', 'or_booking', 'ar_booking']
inventory_use_cols = ['STORE_NM','STYLCOLOR_CD','EOP_QTY', 'EOP_DT','IN_TRANSIT_QTY', 'bz_wh_transit_qty']

platform_dict = {
'L&F S-SDC DIGITAL': "NIKE.COM", 
'Tmall': "TMALL", 
'Nike.com': "NIKE.COM", 
'NIKE.COM-NORTH': "NIKE.COM",
'TMALL Jordan': "TMALL", 
'TMALL YA': "TMALL", 
'GROUP PURCHASE': "NIKE.COM"}

buy_plan_channel_define = {
'CN FY11 E-Commerce':'TMALL',
'Store.com':'NIKE.COM',
'CN Digital Tmall':'TMALL', 
'GROUP PURCHASE':'NIKE.COM', 
'TMALL STORE-JORDAN': 'TMALL', 
'TMALL STORE-YA': 'TMALL'}


final_dataset_have_cols = ['platform', 'style_cd', 'week_end', 'week_begin', 'YrWkNum',
                           'sales_qty', 'sales_amt', 'msrp_amt', 'STYLCOLOR_DESC', 'gender',
                           'category', 'PE', 'MSRP', 'Traffic', 'EOP_QTY', 'common_style_su19',
                           'DaysRM', 'DaysRM_PreheatIncluded', 'season_buy_qty',
                           'style_traffic_week', 'style_pv_week', 'avg_selling_price', 'sn_wk0',
                           'sn_wk_end', 'sn_max_traffic', 'sn_max_inventory', 'md', 'sales_wk',
                           'q0', 'max_wk_sold', 'median_wk_sold', 'avg_wk_sold']

season_periods_dict = {"SP": ("01-01","03-31"),
                       "SU": ("03-31","07-01"),
                       "FA": ("07-01","10-01"),
                       "HO": ("10-01", "12-31"),
                       "Jan": ("01-01", "02-01"),
                       "Feb": ("02-01", "03-01"),
                       "Mar": ("03-01", "04-01")}

agg_dict = {
    'sales_qty':sum,
    'sales_amt':sum,
    "msrp_amt": sum,
    'STYLCOLOR_DESC':'first',
    'gender':'first',
    'category':'first',
    'PE':'first',
    'MSRP': pd.Series.median,
    'Traffic':'mean',
    'EOP_QTY': sum}

agg_model_dict = {
    'sales_qty':sum,
    'sales_amt':sum,
    "msrp_amt": sum,
    'STYLCOLOR_DESC':'first',
    'gender':'first',
    'category':'first',
    'PE':'first',
    'MSRP': pd.Series.median,
    'Traffic':'mean',
    'EOP_QTY': sum,
    'DaysRM': 'mean',
    'DaysRM_PreheatIncluded': 'mean',
    'season_buy_qty': sum,
    'style_traffic_week': sum,
    'style_pv_week': sum}


class PathManager:
    def __init__(self, base_url):
        self.base_url = base_url

    def load_paths(self, path_dict):
        "path_dict: path dict location or paths dictionary"
        if not isinstance(path_dict, dict):
            path_dict = json.load(open(os.path.join(path_dict, "paths_config.json")))
        for name, value in path_dict.items():
            setattr(self, name, os.path.join(self.base_url, value))
            if not os.path.exists(os.path.join(self.base_url, value)):
                os.makedirs(os.path.join(self.base_url, value))

    def add_source(self,name,path):
        setattr(self,name,os.path.join(self.base_url, path))
        if not os.path.exists(os.path.join(self.base_url, path)):
            os.makedirs(os.path.join(self.base_url, path))

    def save_paths(self, location):
        path_dict_new = {}
        all_attr = dir(self)
        all_attr = [x for x in all_attr if not x.startswith("__") and x!='base_url']
        for attr in all_attr:
            if not inspect.ismethod(getattr(self, attr)):
                path_dict_new[attr] = getattr(self, attr)
        with open(os.path.join(location, "paths_config.json"),'w') as f:
            json.dump(path_dict_new, f)


path_dict = {"source":"source_data", "inseason_analysis":"inseason_analysis", "step_data":"step_data"}
paths = PathManager("")
paths.load_paths(path_dict)


###### inseason md modeling
season_start_dict = {
"SP2018": "2018-01-01", "SP2019": "2019-01-01", "SP2020": "2020-01-01",
"SU2018": "2018-04-01", "SU2019": "2019-04-01", "SU2020": "2020-04-01",
"FA2018": "2018-07-01", "FA2019": "2019-07-01", "FA2020": "2020-07-01",
"HO2018": "2018-10-01", "HO2019": "2019-10-01", "HO2020": "2020-10-01"}

season_end_dict = {
"SP2018": "2018-03-31", "SP2019": "2019-03-31", "SP2020": "2020-03-31",
"SU2018": "2018-06-30", "SU2019": "2019-06-30",
"FA2018": "2018-09-30", "FA2019": "2019-09-30",
"HO2018": "2018-12-31", "HO2019": "2019-12-31"}

func_dict = {"mean": np.mean, "std": np.std, "max": np.max, "min": np.min, "median": np.median, "sum": np.sum}

sophie_features = [
 "sales_date",
 "prod_code",
 'AUR_inv_adv',
 'AUR_inv_adv_last_md',
 'aps_last_md',
 'first_14d_APS',
 'first_14d_avg_md',
 'first_sales_date', 
 'inseason_1st_day',
 'inseason_APS_this_md',
 'interval_after_last_md',
 'inv_adv',
 'inv_adv_last_md',
 'md_inv_adv',
 'md_inv_adv_last_md',
 'num_days_comp_this_md_inseason',
 'num_days_on_mkt',
 'num_days_on_mkt_inseason',
 'num_days_this_md_inseason',
 'num_inseasons',
 'num_seasons',
 'num_styles_in_group',
 'num_styles_lower_MD_in_group',
 'num_styles_on_sale_in_group',
 'this_md_first_day']

competing_features_cols = ['same__MSRP_md_competitiveness',
                           'same__MSRP_comp_cnt',
                           'same__MSRP_AUR_rank',
                           'same__AUR_MSRP_competitiveness',
                           'same__AUR_comp_cnt',
                           'same__AUR_past_7_aps_rank',
                           'same_platform_AUR_MSRP_competitiveness',
                           'same_platform_AUR_comp_cnt',
                           'same_platform_AUR_past_7_aps_rank',
                           'same_pc_AUR_MSRP_competitiveness',
                           'same_pc_AUR_comp_cnt',
                           'same_pc_AUR_past_7_aps_rank',
                           'same_Model_AUR_MSRP_competitiveness',
                           'same_Model_AUR_comp_cnt',
                           'same_Model_AUR_past_7_aps_rank']

competing_features_cols_sku =  competing_features_cols + ['same_color_AUR_MSRP_competitiveness',
                                                          'same_color_AUR_comp_cnt',
                                                          'same_color_AUR_past_7_aps_rank']

md_sensitivity_model_master_file = "md_sensitivity_modeling_data_master_99_rm.csv"
md_sensitivity_model_traffic_file = "TRAFFIC_site_most_recent_daily.xlsx"
competing_styles_existing_features = "competing_styles_features_to_date.csv"
competing_skus_existing_features = "competing_skus_features_to_date.csv"
sku_color_features_to_date = "sku_color_features_to_date.csv"

# sophie's features
# sophie_features_src = "new_data_src20200428/from_sophie/20200810df_style_99plan_A.csv"
# sophie_features_sku_src = "new_data_src20200428/from_sophie/20200810df_sku_99plan_A.csv"

sophie_features_src = "new_data_src20200428/from_sophie/20200812df_style_99plan_B.csv"
sophie_features_sku_src = "new_data_src20200428/from_sophie/20200812df_sku_99plan_B.csv"

# Ruofei features
ruofei_feature_src = "tmall_sku_daily_data_master_fcst.csv"

# zhaoxu's retail moment features
zhaoxu_sku_features = 'source_data/new_data_src20200428/from_zhaoxu/sku_feature_0718.csv'
rm_calendar = '2017_2020_rm_notes.xlsx'             # manually maintained
zx_rm_sku_features = 'new_data_src20200428/from_zhaoxu/sku_rm_features_yzx.csv'
zx_rm_style_features = 'new_data_src20200428/from_zhaoxu/style_rm_features_yzx.csv'
local_retail_moment_only_data_src = "2017_2020_retail_moments_src_data.csv"

cleaned_color_words = "color_words_used.xlsx"
local_inventory_db = "inventory_data.csv"

data_master_include_forecast_day = "fcst_20200909.csv"
data_master_include_forecast_day_B = "fcst_20200909_B.csv"