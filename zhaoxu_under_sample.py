raw=pd.read_csv(r'C:\Users\SYan87\Documents\sku_model\style_level\md_sensitivity_modeling_data_master_99_rm.csv')
raw=raw.loc[raw.md<1]
len(raw.columns)
raw.dropna(subset=['past_3_median_aps'],inplace=True)
df=raw.copy()
df['sales_date']=pd.to_datetime(df['sales_date'])

#读取rm_date 表格
date_df=pd.read_excel(r'C:\Users\SYan87\Documents\sku_model\traffic\style_level_rm_date.xlsx') 
date_df['sales_date']=pd.to_datetime(date_df['sales_date'])
date=date_df['sales_date'].values

#创建label

df['is_rm_2']=0
df.loc[df.sales_date.isin(date),'is_rm_2']=1

train_1=df.query("((sales_date>'2018-06-30')&(sales_date<'2019-01-01')|(sales_date>'2019-03-31')&(sales_date<'2019-07-01'))").drop(['sales_qty','past_3_median_aps','sales_date'],axis=1)
train_2=df.query("(sales_date>'2019-06-30')&(sales_date<'2019-08-26')").drop(['sales_qty','past_3_median_aps','sales_date'],axis=1)
test=df.query("(sales_date=='2019-09-09')").drop(['sales_qty','past_3_median_aps','sales_date'],axis=1)

#下采样

from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(sampling_strategy=0.3,random_state=0)
X_resampled, y_resampled = rus.fit_sample(train_1.drop('is_rm_2',axis=1), train_1['is_rm_2'])

y_resampled=pd.DataFrame(y_resampled)
train_1=pd.concat([X_resampled,y_resampled],axis=1)

train=pd.concat([train_1,train_2],axis=0)
X_train=train.drop('y_data',axis=1)
X_test=test.drop('y_data',axis=1)
y_train=train['y_data']
y_test=test['y_data']