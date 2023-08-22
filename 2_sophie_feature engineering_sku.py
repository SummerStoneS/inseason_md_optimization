#####2020.6.30
import sys
import os
import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
from config import *


#######1. parameters setting (NEED UPDATE EVERY TIME)

# pass_days=14   # 提前多少天预测， already defined in config.py
code_start_date = "2020-07-07"                  # 只对增量数据做feature日期开始时间
code_end_date="2020-07-20"
fcst_date = "2020-07-20"                # if not exist use "" or any date not in data
excl_rm = 'N'    # exclue retail moments
PE = 'FOOTWEAR DIVISION'
# is_forecast = 'N'

# load latest version of feature data
df_old = pd.read_csv(os.path.join(paths.source, sophie_features_sku_src))

# load raw data from csv
sold = pd.read_csv(os.path.join(paths.step_data, data_master_include_forecast_day))

is_forecast='N'

######################################
######################################

calendar = pd.read_excel(f"{paths.source}/{calendar_file}")

season_start_list=calendar.drop_duplicates(['Season'], 'first', inplace=False)[['DATE', 'Season']]

season_end_list=calendar.drop_duplicates(['Season'], 'last', inplace=False)[['DATE', 'Season']]

pass_days=datetime.timedelta(pass_days)


if excl_rm == 'N':
# retail moment calendar
    rm_dates = []

df=sold.copy()

# footwear
df=df[df["Division"] == PE]

# convert to datetime
df['sales_date'] = pd.to_datetime(df['sales_date'])
df['First Offer Date'] = pd.to_datetime(df['First Offer Date'])

# take the data ended on code_end_date
df=df.query("(sales_date<=@code_end_date)")

# rename Style Number and GC Category
df.rename(columns = {'Style Number': 'Style_Number','GC Category': 'GC_Category'}, inplace = True)

# Find style's Gender group and GC_Category based on most sold products
df=df.sort_values(axis = 0,ascending = [True,True,False],by = ['sales_date','prod_code','sales_amt'])[['sales_date', 'prod_code', 'Style_Number','inseason_flag1', 'sales_qty',
       'sales_amt', 'MSRP', 'inv_qty', 'md', 'Gender Group', 'GC_Category',
       'season1', 'SEASON', 'Subcategory']]

# 2020.7.8 calculate markdown price and discount
df["md_price"] = np.where(df["sales_date"].isin([fcst_date]), df.eval("MSRP*md"), df.eval("sales_amt/sales_qty"))
df["md"] = df["md"].round(1)

# if is_forecast=='N':
#     # calculate markdown price and discount
#     df['md_price']=df['sales_amt']/df['sales_qty']
#     df['md']=df['md_price']/df['MSRP']
#     df['md']=round(df['md'],1)
# else:
#     df['md_price']=df.eval("MSRP*md")
#     df['md']=round(df['md'],1)

# PRICE BAND CANNIBALIZATION  <2000 +/- 100  >2000 +/- 300
df['comp_price_low']=df.apply(lambda x: x['md_price']-100 if x['md_price'] < 2000 else x['md_price']-300, axis=1)
df['comp_price_high']=df.apply(lambda x: x['md_price']+100 if x['md_price'] < 2000 else x['md_price']+300, axis=1)


###############################################################
# number of seasons from first season launched till now 

season_list=df.query("(sales_date>=@code_start_date)&(sales_date<=@code_end_date)").season1.unique()

season_number = pd.DataFrame()

for i in range(len(season_list)):
    print(season_list[i])
    season_end=pd.to_datetime(season_end_list[season_end_list["Season"]==season_list[i]]["DATE"].iloc[0])
    season_number_i=df.query("sales_date<=@season_end").groupby(['prod_code']).agg({'season1':pd.Series.nunique}).reset_index()
    season_number_i.rename(columns = {'season1': 'num_seasons'}, inplace = True)
    season_number_i['season1']=season_list[i]
    season_number=season_number.append(season_number_i)

df = pd.merge(df, season_number, on = ['prod_code','season1'], how = 'left')




###############################################################
# of inseason seasons experienced
inseason_number = pd.DataFrame()

for i in range(len(season_list)):
    print(season_list[i])
    season_end=pd.to_datetime(season_end_list[season_end_list["Season"]==season_list[i]]["DATE"].iloc[0])
    inseason_number_i=df.query("(sales_date<=@season_end)&(inseason_flag1=='Y')").groupby(['prod_code']).agg({'season1':pd.Series.nunique}).reset_index()
    inseason_number_i.rename(columns = {'season1': 'num_inseasons'}, inplace = True)
    inseason_number_i['season1']=season_list[i]
    inseason_number=inseason_number.append(inseason_number_i)

df = pd.merge(df, inseason_number, on = ['prod_code','season1'], how = 'left')



###############################################################
# number of styles within the same group of pe/gender/GC_Category/AUR
# number of styles on sale within the same group of pe/gender/GC_Category/AUR
# number of styles at a lower discount within the same group of pe/gender/GC_Category/AUR


def cal_styles(a,b):
    # a=pd.to_datetime(season_start_dict[season])-datetime.timedelta(days=14)
    # b=pd.to_datetime(season_end_dict[season])

    data=df.query("(sales_date<=@b)&(sales_date>=@a)")[['sales_date','season1','md_price','Style_Number','md','comp_price_low','comp_price_high','group','num_styles_in_group','num_styles_on_sale_in_group','num_styles_lower_MD_in_group']]
    data=data.sort_values(axis = 0,ascending = [True,True,False],by = ['sales_date','Style_Number','md'])
    data=data.drop_duplicates(['Style_Number','sales_date'], 'last', inplace=False)
    for x in range(0,data.shape[0]): 
        x_date1=data.iloc[x,0]
        x_group=data.iloc[x,7]
        x_low=data.iloc[x,5]
        x_high=data.iloc[x,6] 
        x_md=data.iloc[x,4] 
        y1=data.query("(sales_date==@x_date1)&(group==@x_group)&(md_price>=@x_low)&(md_price<=@x_high)").drop_duplicates(['Style_Number'], 'last', inplace=False)
        data.iloc[x,8]=y1.shape[0]      
        y2=data.query("(sales_date==@x_date1)&(md<1)&(group==@x_group)&(md_price>=@x_low)&(md_price<=@x_high)").drop_duplicates(['Style_Number'], 'last', inplace=False)
        data.iloc[x,9]=y2.shape[0]
        y3=data.query("(sales_date==@x_date1)&(md<@x_md)&(group==@x_group)&(md_price>=@x_low)&(md_price<=@x_high)").drop_duplicates(['Style_Number'], 'last', inplace=False)
        data.iloc[x,10]=y3.shape[0]
        print(y1)
        print(x_date1)

    return data


df["group"] = df.apply(lambda x: x["GC_Category"]+ "_" +x['Gender Group'], axis=1)
df['num_styles_in_group']=0
df['num_styles_on_sale_in_group']=0
df['num_styles_lower_MD_in_group']=0

num_styles=cal_styles(code_start_date,code_end_date)
num_styles=num_styles[['sales_date','Style_Number','num_styles_in_group','num_styles_on_sale_in_group','num_styles_lower_MD_in_group']]



df=df.drop(["num_styles_in_group","num_styles_on_sale_in_group","num_styles_lower_MD_in_group"],axis=1)
df = pd.merge(df, num_styles, on = ['Style_Number','sales_date'], how = 'left')


###############################################################
# first sales day of this season


inseason_1st_day = pd.DataFrame()

for i in range(len(season_list)):
    print(season_list[i])
    season_start=pd.to_datetime(season_start_list[season_start_list["Season"]==season_list[i]]["DATE"].iloc[0])
    season_end=pd.to_datetime(season_end_list[season_end_list["Season"]==season_list[i]]["DATE"].iloc[0])
    inseason_1st_day_i=df.query("(sales_date<=@season_end)&(sales_date>=@season_start)")[['season1','prod_code','sales_date']]
    inseason_1st_day_i=inseason_1st_day_i.drop_duplicates(['prod_code'], 'first', inplace=False)[['prod_code','sales_date','season1']]
    inseason_1st_day_i.rename(columns = {'sales_date': 'inseason_1st_day'}, inplace = True)
    inseason_1st_day=inseason_1st_day.append(inseason_1st_day_i)

df = pd.merge(df, inseason_1st_day, on = ['prod_code','season1'], how = 'left')
    



###############################################################
# average APS of first 14 days of sales

# calculate the first sales date after 2018-01-01
temp=df.copy()
temp=temp.drop_duplicates(['prod_code'], 'first', inplace=False)
temp.rename(columns = {'sales_date': 'first_sales_date'}, inplace = True)
temp=temp[['first_sales_date','prod_code']]
df = pd.merge(df, temp, on = ['prod_code'], how = 'left')


first_14d = pd.DataFrame()

for i in range(len(season_list)):
    print(season_list[i])

    a=pd.to_datetime(season_start_list[season_start_list["Season"]==season_list[i]]["DATE"].iloc[0])
    ##season start+13 days
    a14=a+datetime.timedelta(days=13)
    
    
    first_14d_i=df.query("(sales_date<=@a14)&(sales_date>=@a)")[['season1','prod_code','sales_qty','md','inseason_1st_day']]
    first_14d_i['days_after_inseason_1st_day']=(a+datetime.timedelta(days=14)-first_14d_i['inseason_1st_day'])/datetime.timedelta(days=1)
    first_14d_i["sales_qty"] = first_14d_i.eval("sales_qty/days_after_inseason_1st_day")
    print(first_14d_i)
    
    first_14d_i=first_14d_i.groupby(['prod_code','season1']).agg({'sales_qty':'sum','md':'mean'}).reset_index()
    first_14d_i.rename(columns = {'sales_qty': 'first_14d_APS','md': 'first_14d_avg_md'}, inplace = True)

    first_14d=first_14d.append(first_14d_i)


df = pd.merge(df, first_14d, on = ['prod_code','season1'], how = 'left')



for i in range(len(season_list)):

    a=pd.to_datetime(season_start_list[season_start_list["Season"]==season_list[i]]["DATE"].iloc[0])
    ##season start+13 days
    a14=a+datetime.timedelta(days=13)
    
    # remove data during the first 14 days after season starts and gap days
    df["first_14d_APS"] = np.where(((df["sales_date"] >= a) & (df["sales_date"] <= a14+pass_days)) , np.nan, df["first_14d_APS"])
    df["first_14d_avg_md"] = np.where(((df["sales_date"] >= a) & (df["sales_date"] <= a14+pass_days)) , np.nan, df["first_14d_avg_md"])

###############################################################
# first day in this season under this markdown

md_first_day = pd.DataFrame()

for i in range(len(season_list)):
    print(season_list[i])
    season_start=pd.to_datetime(season_start_list[season_start_list["Season"]==season_list[i]]["DATE"].iloc[0])
    season_end=pd.to_datetime(season_end_list[season_end_list["Season"]==season_list[i]]["DATE"].iloc[0])

    md_first_day_i=df.query("(sales_date<=@season_end)&(sales_date>=@season_start)").drop_duplicates(['prod_code','md'], 'first', inplace=False)[['sales_date','prod_code','md','season1']]
    md_first_day_i.rename(columns = {'sales_date': 'this_md_first_day'}, inplace = True)
    md_first_day=md_first_day.append(md_first_day_i)


df = pd.merge(df, md_first_day, on = ['prod_code','md','season1'], how = 'left')




###############################################################
# aps in this season under this markdown

df['inseason_APS_this_md']=np.nan

inseason_APS_this_md = pd.DataFrame()


for i in range(len(season_list)):
    print(season_list[i])
    season_start=pd.to_datetime(season_start_list[season_start_list["Season"]==season_list[i]]["DATE"].iloc[0])
    season_end=pd.to_datetime(season_end_list[season_end_list["Season"]==season_list[i]]["DATE"].iloc[0])

# 真正需要运算的日期
    inseason_APS_this_md_i=df.query("(sales_date<=@season_end)&(sales_date>=@season_start)&(sales_date>=@code_start_date)")[['sales_date','prod_code','sales_qty','season1','md','inseason_APS_this_md']]

# 取数的data
    data=df.query("(sales_date<=@season_end)&(sales_date>=@season_start)")[['sales_date','prod_code','sales_qty','season1','md','inseason_APS_this_md']]

    #delete retail moments
    data=data[~data.sales_date.isin(rm_dates)]


    print(inseason_APS_this_md_i.shape[0])
    for x in range(0,inseason_APS_this_md_i.shape[0]): 
        x_date1=inseason_APS_this_md_i.iloc[x,0]-pass_days
        x_md=inseason_APS_this_md_i.iloc[x,4]
        x_prod_code=inseason_APS_this_md_i.iloc[x,1]
        temp=data.query("(sales_date<=@x_date1)&(md==@x_md)&(prod_code==@x_prod_code)")
        inseason_APS_this_md_i.iloc[x,5]=temp.sales_qty.mean()
    
    inseason_APS_this_md=inseason_APS_this_md.append(inseason_APS_this_md_i)


df=df.drop(["inseason_APS_this_md"],axis=1)
inseason_APS_this_md=inseason_APS_this_md[['prod_code','sales_date','md','inseason_APS_this_md']]
df = pd.merge(df, inseason_APS_this_md, on = ['prod_code','sales_date','md'], how = 'left')




###############################################################
# number of on-market days 
# number of on-market days in season

df2=df.copy()

df['num_days_on_mkt']=np.nan
df['num_days_on_mkt_inseason']=np.nan



def cal_on_market_days(a,b):

    data=df.query("(sales_date<=@b)&(sales_date>=@a)")[['sales_date','prod_code','num_days_on_mkt','num_days_on_mkt_inseason']]
    
    for x in range(0,data.shape[0]): 
        x_date1=data.iloc[x,0]
        x_prod_code=data.iloc[x,1]
        temp=df.query("(sales_date<=@x_date1)&(prod_code==@x_prod_code)")
        data.iloc[x,2]=temp.shape[0]
        data.iloc[x,3]=temp.query("inseason_flag1=='Y'").shape[0]
        print(x)        
        
    return data



num_days_on_mkt=cal_on_market_days(code_start_date,code_end_date)

df=df.drop(["num_days_on_mkt","num_days_on_mkt_inseason"],axis=1)
df = pd.merge(df, num_days_on_mkt, on = ['prod_code','sales_date'], how = 'left')


###############################################################
# inventory of this product/ sum of inventory of products within the same group of pe/gender/GC_Category/AUR
# AUR * inventory of this product/ AUR* inventory of other styles within the same group
# md * inventory of this product/ AUR* inventory of other styles within the same group

df3=df.copy()

def cal_AUR_inv_advantage(a,b):
    # a=pd.to_datetime(season_start_dict[season])
    # b=pd.to_datetime(season_end_dict[season])
    
    data=df.query("(sales_date<=@b)&(sales_date>=@a)")[['sales_date','prod_code','inv_qty','group','md_price','inv_adv','AUR_inv_adv','md','md_inv_adv','comp_price_low','comp_price_high']]
  
    
    for x in range(0,data.shape[0]): 
        x_date1=data.iloc[x,0]-pass_days
        x_group=data.iloc[x,3]
        x_low=data.iloc[x,9]
        x_high=data.iloc[x,10]
        x_prod_code=data.iloc[x,1]
        
        data2=data.query("(sales_date==@x_date1)&(prod_code==@x_prod_code)")
        
        temp=data.query("(sales_date==@x_date1)&(group==@x_group)&(md_price>=@x_low)&(md_price<=@x_high)")
        temp['aur_inv']=temp.eval("md_price*inv_qty")
        temp['md_inv']=temp.eval("md*inv_qty")
        
        if (temp['inv_qty'].sum()>0)&(data2.shape[0]==1):
            data.iloc[x,5]=data2.iloc[0,2]/temp['inv_qty'].sum()
            data.iloc[x,6]=data2.iloc[0,2]*data2.iloc[0,4]/(temp['aur_inv'].sum())
            data.iloc[x,8]=data2.iloc[0,7]*data2.iloc[0,4]/(temp['md_inv'].sum())
            
        print(x)
    return data

df['inv_adv']=np.nan
df['AUR_inv_adv']=np.nan
df['md_inv_adv']=np.nan

inv_adv=cal_AUR_inv_advantage(code_start_date,code_end_date)
inv_adv=inv_adv[['sales_date','prod_code','inv_adv','AUR_inv_adv','md_inv_adv']]

df=df.drop(["inv_adv","AUR_inv_adv","md_inv_adv"],axis=1)
df = pd.merge(df, inv_adv, on = ['prod_code','sales_date'], how = 'left')



###############################################################
# last time under this markdown's aps
# days after last time's markdown
# last time under this md - inventory of this product/ sum of inventory of products within the same group of pe/gender/GC_Category/AUR
# last time under this md - AUR * inventory of this product/ AUR* inventory of other styles within the same group
# last time under this md - md * inventory of this product/ AUR* inventory of other styles within the same group

df4=df.copy()


def cal_last_md(a,b):

    data=df.query("(sales_date<=@b)&(sales_date>=@a)")[['sales_date','prod_code','md','aps_last_md','interval_after_last_md','inv_qty','group','md_price','inv_adv_last_md','AUR_inv_adv_last_md','md_inv_adv_last_md','sales_qty','comp_price_low','comp_price_high']]
  
    
    for x in range(0,data.shape[0]): 
        x_date1=data.iloc[x,0]-pass_days
        x_prod_code=data.iloc[x,1]
        x_md=data.iloc[x,2]
        x_group=data.iloc[x,6]
        x_low=data.iloc[x,12]
        x_high=data.iloc[x,13]

        temp=df.query("(sales_date<=@x_date1)&(md==@x_md)&(prod_code==@x_prod_code)")
        #delete retail moments or not based on excl_rm
        temp=temp[~temp.sales_date.isin(rm_dates)]

        temp=temp.drop_duplicates(['prod_code','md'], 'last', inplace=False)
        

        if (temp.shape[0]==1):
            data.iloc[x,3]=temp['sales_qty'].iloc[0]
            data.iloc[x,4]=(x_date1-temp['sales_date'].iloc[0])/datetime.timedelta(days=1)
            temp_date=temp['sales_date'].iloc[0]
            temp2=df.query("(sales_date==@temp_date)&(group==@x_group)&(md_price>=@x_low)&(md_price<=@x_high)")
            
            temp2['aur_inv']=temp2.eval("md_price*inv_qty")
            temp2['md_inv']=temp2.eval("md*inv_qty")
            
            if temp2['inv_qty'].sum()>0:
                data.iloc[x,8]=temp['inv_qty'].iloc[0]/(temp2['inv_qty'].sum())
                data.iloc[x,9]=temp['inv_qty'].iloc[0]*temp['md_price'].iloc[0]/(temp2['aur_inv'].sum())
                data.iloc[x,10]=temp['inv_qty'].iloc[0]*temp['md'].iloc[0]/(temp2['md_inv'].sum())
            print(x)

            
    return data

df['aps_last_md']=np.nan
df['interval_after_last_md']=np.nan
df['inv_adv_last_md']=np.nan
df['AUR_inv_adv_last_md']=np.nan
df['md_inv_adv_last_md']=np.nan


last_md=cal_last_md(code_start_date,code_end_date)
last_md=last_md[['sales_date','prod_code','md','aps_last_md','interval_after_last_md','inv_adv_last_md','AUR_inv_adv_last_md','md_inv_adv_last_md']]

df=df.drop(["aps_last_md","interval_after_last_md","inv_adv_last_md","AUR_inv_adv_last_md","md_inv_adv_last_md"],axis=1)
df = pd.merge(df, last_md, on = ['prod_code','sales_date','md'], how = 'left')


###############################################################
# number of days under this markdown in this season
# number of days comps under this markdown in this season
df5=df.copy()


df['num_days_this_md_inseason']=np.nan
df['num_days_comp_this_md_inseason']=np.nan

days_under_this_markdown = pd.DataFrame()


for i in range(len(season_list)):
    print(season_list[i])
    season_start=pd.to_datetime(season_start_list[season_start_list["Season"]==season_list[i]]["DATE"].iloc[0])
    season_end=pd.to_datetime(season_end_list[season_end_list["Season"]==season_list[i]]["DATE"].iloc[0])

    data=df.query("(sales_date<=@season_end)&(sales_date>=@season_start)")[['sales_date','prod_code','md','num_days_this_md_inseason','num_days_comp_this_md_inseason','group','comp_price_low','comp_price_high','md_price']]


    for x in range(0,data.shape[0]): 
        x_date1=data.iloc[x,0]-pass_days
        x_md=data.iloc[x,2]
        x_prod_code=data.iloc[x,1]
        x_group=data.iloc[x,5]
        x_low=data.iloc[x,6]
        x_high=data.iloc[x,7]       

        temp=data.query("(sales_date<=@x_date1)&(md==@x_md)&(prod_code==@x_prod_code)")
        data.iloc[x,3]=temp.shape[0]
        temp2=data.query("(sales_date<@x_date1)&(md==@x_md)&(group==@x_group)&(prod_code!=@x_prod_code)&(md_price>=@x_low)&(md_price<=@x_high)")
        temp2=temp2.drop_duplicates('sales_date', 'last', inplace=False)
        data.iloc[x,4]=temp2.shape[0]
        print(x)

    days_under_this_markdown=days_under_this_markdown.append(data)


days_under_this_markdown=days_under_this_markdown[['sales_date','prod_code','md','num_days_this_md_inseason','num_days_comp_this_md_inseason']]

df=df.drop(["num_days_this_md_inseason","num_days_comp_this_md_inseason"],axis=1)
df = pd.merge(df, days_under_this_markdown, on = ['prod_code','sales_date','md'], how = 'left')



###########replace the old feature data

df_new=df.query("(sales_date>=@code_start_date)&(sales_date<=@code_end_date)")[['prod_code', 'sales_date', 'inseason_flag1',
        'sales_qty', 'sales_amt', 'MSRP', 'inv_qty', 'Gender Group',
        'GC_Category', 'season1', 'SEASON', 'Subcategory', 'md_price', 'md',
        'comp_price_low', 'comp_price_high', 'num_seasons', 'num_inseasons',
        'group', 'num_styles_in_group', 'num_styles_on_sale_in_group',
        'num_styles_lower_MD_in_group', 'inseason_1st_day', 'first_sales_date',
        'first_14d_APS', 'first_14d_avg_md', 'this_md_first_day',
        'inseason_APS_this_md', 'num_days_on_mkt', 'num_days_on_mkt_inseason',
        'inv_adv', 'AUR_inv_adv', 'md_inv_adv', 'aps_last_md',
        'interval_after_last_md', 'inv_adv_last_md', 'AUR_inv_adv_last_md',
        'md_inv_adv_last_md', 'num_days_this_md_inseason',
        'num_days_comp_this_md_inseason']]

df_old['sales_date'] = pd.to_datetime(df_old['sales_date'])

df_old=df_old.query("(sales_date<@code_start_date)|(sales_date>@code_end_date)")[['prod_code', 'sales_date', 'inseason_flag1',
        'sales_qty', 'sales_amt', 'MSRP', 'inv_qty', 'Gender Group',
        'GC_Category', 'season1', 'SEASON', 'Subcategory', 'md_price', 'md',
        'comp_price_low', 'comp_price_high', 'num_seasons', 'num_inseasons',
        'group', 'num_styles_in_group', 'num_styles_on_sale_in_group',
        'num_styles_lower_MD_in_group', 'inseason_1st_day', 'first_sales_date',
        'first_14d_APS', 'first_14d_avg_md', 'this_md_first_day',
        'inseason_APS_this_md', 'num_days_on_mkt', 'num_days_on_mkt_inseason',
        'inv_adv', 'AUR_inv_adv', 'md_inv_adv', 'aps_last_md',
        'interval_after_last_md', 'inv_adv_last_md', 'AUR_inv_adv_last_md',
        'md_inv_adv_last_md', 'num_days_this_md_inseason',
        'num_days_comp_this_md_inseason']]


df=df_old.append(df_new)


df=df.sort_values(axis = 0,ascending = [True,True,False],by = ['sales_date','prod_code','sales_amt'])

df.to_csv(os.path.join(paths.source, sophie_features_sku_src))



