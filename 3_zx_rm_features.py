# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 15:37:17 2020

@author: SYan87
"""

import pandas as pd
import os
import datetime
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from config import *


  
def last_rm(rm):
    last_rm=rm.groupby(level_col+['Gender_Group','number']).agg({'rm_sales_qty':'sum','rm_md':'mean','sales_date':'max','rm_qty_rank':'max','rm_qty_inv_adv':'max'}).reset_index()    
    last_rm.rename(columns={'number':'last_number','rm_sales_qty':'last_rm_sales_qty','rm_md':'last_rm_md','sales_date':'sku_in_last_rm_max_date',
                            'rm_qty_rank':'last_rm_qty_rank','rm_qty_inv_adv':'last_rm_qty_inv_adv'},inplace=True)
    rm['last_number']=rm['number']-1
    rm=pd.merge(rm,last_rm,on=level_col+['last_number','Gender_Group'],how='left')
    rm['last_rm_interval']=(rm['sales_date']-rm['sku_in_last_rm_max_date'])/datetime.timedelta(days=1)
    rm['last_rm_flag']=rm['last_rm_sales_qty'].apply(lambda x:1 if x>0 else 0)
    return rm


def last_same_rm(rm):
    # 去年同款大促第一天的销量，非第一天的平均销量和去年同款大促的平均md
    rm['last_year']=rm['year']-1
    rm_1=rm.loc[rm.day_number==1]
    rm_2=rm.loc[rm.day_number>1]

    last_same_rm_qty_1=rm_1.groupby(level_col+['RM_name','year']).agg({'rm_sales_qty':'sum'}).reset_index().rename(columns={'year':'last_year','rm_sales_qty':'last_same_rm_qty'})
    last_same_rm_qty_2=rm_2.groupby(level_col+['RM_name','year']).agg({'rm_sales_qty':'mean'}).reset_index().rename(columns={'year':'last_year','rm_sales_qty':'last_same_rm_qty'})
    last_same_rm_md=rm.groupby(level_col+['RM_name','year']).agg({'rm_md':'mean'}).reset_index().rename(columns={'year':'last_year','rm_md':'last_same_rm_md'})
    
    rm=pd.merge(rm_1,last_same_rm_qty_1,on=level_col+['RM_name','last_year'],how='left').append(pd.merge(rm_2,last_same_rm_qty_2,on=level_col+['RM_name','last_year'],how='left'))
    rm=pd.merge(rm,last_same_rm_md,on=level_col+['RM_name','last_year'],how='left')
    rm['last_same_rm_flag']=rm['last_same_rm_qty'].apply(lambda x:1 if x>0 else 0)
    return rm


def hist_rm(rm):
    feas=['hist_rm_count','hist_rm_qty','hist_rm_md','hist_top_10_flag','hist_hot_flag','hist_hot_interval']
    for column in feas:
        rm[column] = np.nan
    for x in range(len(rm)):
        x_number=rm['number'].iloc[x]
        x_code=rm[level_col[0]].iloc[x]
        x_sales_date=rm['sales_date'].iloc[x]
        if level=='style':
            temp=rm.query("(style_cd==@x_code)&(number<@x_number)")
        else:
            temp=rm.query("(prod_code==@x_code)&(number<@x_number)")
        rm['hist_rm_count'].iloc[x]=temp.number.nunique()
        rm['hist_rm_qty'].iloc[x]=np.mean(temp.groupby('number')['rm_sales_qty'].agg('sum'))
        rm['hist_rm_md'].iloc[x]=temp.rm_md.mean()
        rm['hist_top_10_flag'].iloc[x]=temp['rm_top_10_flag'].max()
        rm['hist_hot_flag'].iloc[x]=temp['rm_qty_inv_adv'].apply(lambda x: 1 if x>1 else 0).max()
        rm['hist_hot_interval'].iloc[x]=(x_sales_date-temp.loc[temp.hist_hot_flag==1,'sales_date'].max())/datetime.timedelta(days=1)
    return rm


# read retail_moment data
def rm_data_prepare(raw,retail,retail_date,inv):
    # read rm data 
    #retail=pd.read_csv(r'C:\Users\SYan87\Documents\sku_model\feature\2017_2020_retail_moments_src_data(1).csv')
    retail=retail.loc[retail.platform=='TMALL']
    retail['sales_date']=pd.to_datetime(retail['sales_date'])
    retail['style_cd']=retail['prod_code'].apply(lambda x: x[:6])
    
    # select footwear and get gender
    #raw=pd.read_csv(r'C:\Users\SYan87\Documents\sku_model\model_impact_analysis\fcst_20200720.csv')
    division=raw[['Style Number','Division','Gender Group']].drop_duplicates().rename(columns={'Style Number':'style_cd','Gender Group':'Gender_Group'})
    retail=pd.merge(retail,division,how='left',on=['style_cd'])
    retail=retail.loc[retail.Division==PE]
    retail=retail.groupby(level_col+['sales_date','Gender_Group']).agg({'sales_qty':'sum','sales_amt':'sum','MSRP':pd.Series.median}).reset_index()
    
    # read rm dates
    #retail_date=pd.read_excel(r'C:\Users\SYan87\Documents\sku_model\feature\2017_2020_rm_notes.xlsx')
    retail_date['sales_date']=pd.to_datetime(retail_date['sales_date'])
    retail=pd.merge(retail,retail_date,on=['sales_date'],how='left')
    
    # read inventory data
    #inv=pd.read_csv(r'C:\Users\SYan87\Documents\sku_model\feature\inventory_data.csv')
    inv=inv.loc[inv.platform=='TMALL']
    inv["sales_date"] = pd.to_datetime(inv["EOP_DT"]) + datetime.timedelta(days=1)
    inv.rename(columns={"STYLCOLOR_CD":"prod_code", "EOP_QTY":"rm_inv_qty"}, inplace=True)
    inv['style_cd']=inv['prod_code'].apply(lambda x: x[:6])
    inv=inv.groupby(level_col+['sales_date']).agg({'rm_inv_qty':'sum'}).reset_index()
    retail=pd.merge(retail,inv[level_col+['sales_date','rm_inv_qty']],on=level_col+['sales_date'],how='left')
      
    rm=retail.copy()
    rm['rm_md']=round(rm['sales_amt']/rm['sales_qty']/rm['MSRP'],1)
    rm['year']=rm['sales_date'].dt.year
    rm.rename(columns={'sales_qty':'rm_sales_qty','sales_amt':'rm_sales_amt'},inplace=True)
    
    #对大促中同PE,GENDER的sales_qty进行排序输出百分比rank
    rm_sum=rm.groupby(level_col+['Gender_Group','number']).agg({'rm_sales_qty':'sum'}).reset_index().rename(columns={'rm_sales_qty':'sum_sales_qty'})
    rm_qty_rank=rm_sum.set_index(level_col+['number']).groupby(['Gender_Group','number'])['sum_sales_qty'].agg('rank',pct=True,ascending=False).reset_index().rename(columns={'sum_sales_qty':'rm_qty_rank'})
    rm_qty_rank['rm_top_10_flag']=rm_qty_rank['rm_qty_rank'].apply(lambda x: 1 if x<0.1 else 0)
    rm=pd.merge(rm,rm_qty_rank,on=level_col+['number'],how='left')
    
    #计算大促第一天的sales_qty/inv_qty
    first_rm=rm.loc[rm.day_number==1]
    first_rm['rm_qty_inv_adv']=first_rm['rm_sales_qty']/first_rm['rm_inv_qty']
    rm=pd.merge(rm,first_rm[level_col+['number','rm_qty_inv_adv']],on=level_col+['number'],how='left')  
    
    return rm


if __name__=='__main__':
    
     # set parameters
    level = 'sku'  # 'sku' or 'style'
    use_platform = 'TMALL'
    PE = 'FOOTWEAR DIVISION'

    # load data
    retail_moment_sales_src = pd.read_csv(os.path.join(paths.source, local_retail_moment_only_data_src))
    retail_moment_calendar = pd.read_excel(os.path.join(paths.source, rm_calendar))
    src_data = pd.read_csv(f"{paths.step_data}/{data_master_include_forecast_day}")
    inv_data = pd.read_csv(f"{paths.step_data}/{local_inventory_db}")

     # make features
    level_col = ['prod_code'] if level == 'sku' else ['style_cd']
    rm=rm_data_prepare(src_data, retail_moment_sales_src, retail_moment_calendar,inv_data)
    rm_features=last_rm(rm)
    rm_features=last_same_rm(rm_features)
    rm_features=hist_rm(rm_features)    

    features=level_col+['sales_date',
               'last_rm_flag','last_rm_sales_qty','last_rm_md','last_rm_qty_rank','last_rm_qty_inv_adv',
               'last_rm_interval','last_same_rm_flag','last_same_rm_qty',
               'last_same_rm_md','hist_rm_count','hist_rm_qty','hist_rm_md','hist_top_10_flag','hist_hot_flag']
    
 
    rm_features_use=rm_features[features]
    
    if level == "sku":
        rm_features_use.to_csv(os.path.join(paths.source, zx_rm_sku_features),index=0)
    else:
        rm_features_use.to_csv(os.path.join(paths.source, zx_rm_style_features),index=0)
    
