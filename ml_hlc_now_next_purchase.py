#!/usr/bin/env python
# coding: utf-8

# # Step 1: Cleaning Data

# In[128]:


# TENSORFLOW LIB
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Embedding, LSTM
from keras.utils import np_utils
from tensorflow.keras.preprocessing.sequence import pad_sequences
import python_utils
import unidecode

import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling
from keras.callbacks import EarlyStopping


# In[129]:


# import normal LIB
import pandas as pd
import numpy as np
import datetime as datetime
from datetime import date, timedelta, datetime
import csv
from math import floor


# In[130]:


#do not show warnings
import warnings
warnings.filterwarnings("ignore")

#import plotly for visualization
import chart_studio as py
import plotly.offline as pyoff
import plotly.graph_objs as go
import matplotlib.pyplot as plt


# In[131]:


# METRIC AND MODEL VALIDATION 
from sklearn.metrics import roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.cluster import KMeans


# In[132]:


# hàm load data vào
raw_data = pd.read_csv('./ml_now_20200506.csv')
raw_data=raw_data.rename(columns={'partner_id':'customer_id','seito_outlet_id':'store_id','date':'date_order','quantity':'qty','rev':'amount'})
df_covid=pd.read_csv('./vietnam_sumary.csv')
# raw_data = pd.merge(raw_data, df_covid, left_on="dat")


# In[133]:


print('####################  LOAD DATA ######################')
print(raw_data)
print(raw_data.nunique())


# # Custom Daytime data

# In[134]:


# chuyển cột date_order_dt thành datetime bằng pd.to_datetime
raw_data['date_order_dt'] = pd.to_datetime(raw_data['date_order'])
raw_data['date_order_day'] = raw_data['date_order_dt'].dt.strftime('%Y-%m-%d')
raw_data['date_order_day'] = pd.to_datetime(raw_data['date_order_day'])

df_covid['date_report'] = pd.to_datetime(df_covid['date_report'])
# merge with covid
raw_data = pd.merge(raw_data, df_covid, left_on="date_order_day", right_on="date_report", how = "left")
raw_data.head()


# In[135]:


# Fix data bias
raw_data.loc[raw_data.qty < 0, 'qty'] = -raw_data.qty


# In[136]:


raw_data


# In[137]:


# Min date
print('data min date: ',raw_data.date_order_dt.min())
# Max date
print('data max date: ',raw_data.date_order_dt.max())


# In[138]:


# Explore data
df_gby_order = raw_data.groupby(['customer_id']).date_order_day.nunique().reset_index()
df_gby_order.columns = ['customer_id','num_day_order']
df_gby_order.head()


# # Clean data: customer 1 order, mean order per date > 1.7 and order line > 3

# In[139]:


# FIND customer id có 1 ORDER
cus_1_order = df_gby_order[df_gby_order.num_day_order<1.2]
print('FILTER 1** Customer have just 1 order: ', cus_1_order.customer_id.nunique())
# loại bỏ các customer id có số order = 1
raw_data = raw_data[~raw_data.customer_id.isin(cus_1_order.customer_id)]
print('Customer after clean 1 order: ',raw_data.customer_id.nunique())


# # GROUP 1: CUSTOMER have 1 order

# In[140]:


# Calculate order per day
df_gby_num_order = raw_data.groupby(['customer_id']).order_id.nunique().reset_index()
df_gby_num_order.columns = ['customer_id','num_order']
print(df_gby_num_order)
df_gby_num_day_order = raw_data.groupby(['customer_id']).date_order_day.nunique().reset_index()
df_gby_num_day_order.columns = ['customer_id','num_day_order']
df_mean_order_perday = pd.merge(df_gby_num_order, df_gby_num_day_order, how='left', on=['customer_id'])
df_mean_order_perday['mean_of_order_perday'] = df_mean_order_perday['num_order']/df_mean_order_perday['num_day_order']
print('Calculate mean Num order per day: ')
print(df_mean_order_perday)


# In[141]:


print('FILTER 2** customer ID have order per day>1.5 and num_day_order > 30:')
raw_data_cl_mean2orderperday30 = df_mean_order_perday[(df_mean_order_perday.mean_of_order_perday>1.5)&(df_mean_order_perday.num_day_order>30)]
raw_data = raw_data[~raw_data.customer_id.isin(raw_data_cl_mean2orderperday30.customer_id)]
print('Customer unormal: ',raw_data_cl_mean2orderperday30.customer_id.nunique())


# In[142]:


print('FILTER 3** customer have mean_order_perday >1.9')
raw_data_cl_mean2orderperday=df_mean_order_perday[(df_mean_order_perday.mean_of_order_perday>1.9)]
raw_data = raw_data[~raw_data.customer_id.isin(raw_data_cl_mean2orderperday.customer_id)]
print('Customer unormal: ',raw_data_cl_mean2orderperday.customer_id.nunique())

print('FILTER 4** customer buy many quantity per order >25')
filter_qty_per_order = raw_data[raw_data.qty>25]
print('Customer unormal: ',filter_qty_per_order.customer_id.nunique())
#Filter 4 loại bỏ khách hàng có qty_per_order > 25 
raw_data = raw_data[~raw_data.customer_id.isin(filter_qty_per_order.customer_id)]# filter quantity per day
qty_per_day = raw_data.groupby(['customer_id','date_order_day']).qty.sum().reset_index()
qty_per_day.columns = ['customer_id','date_order_d','qty_per_day']
print('FILTER 5** customer buy many quantity per day > 25')
fitler_qty_per_day = qty_per_day[qty_per_day.qty_per_day>25]
print('Customer unormal: ', fitler_qty_per_day.customer_id.nunique())
raw_data = raw_data[~raw_data.customer_id.isin(fitler_qty_per_day.customer_id)]
# # GROUP 4: TOTAL CUSTOMER HAVE BEEN REMOVE

# In[143]:


print(raw_data_cl_mean2orderperday.customer_id.nunique()+raw_data_cl_mean2orderperday30.customer_id.nunique())


# In[144]:


print('Customer left: ',raw_data.nunique())


# # 48220 CUSTOMER Left

# In[145]:


data_h_filter = raw_data.copy()


# # SLIT DATA

# In[146]:


# befor 2019-09-22
df_before = data_h_filter[data_h_filter['date_order_day'] < pd.to_datetime(date(2020, 3, 1))]
df_before = df_before[df_before['date_order_day'] > pd.to_datetime(date(2019, 5, 1))]
df_before


# In[147]:


print('CUT OFF DAY= ', df_before.date_order_day.max())


# In[148]:


# Split data 6 month from max day, cut off day is 2019-09-22
df_after = data_h_filter[data_h_filter['date_order_day'] >= pd.to_datetime(date(2020, 3, 1))]


# # Calculate First Purchase in next 6 month and Last Purchase History

# In[149]:


tx_user_before = pd.DataFrame(df_before['customer_id'].unique())
tx_user_before.columns = ['customer_id']


# In[150]:


#create a dataframe with customer id and first purchase date in next 6 month
tx_next_first_purchase = df_after.groupby('customer_id').date_order_day.min().reset_index()
tx_next_first_purchase.columns = ['customer_id','minpurchaseday']
#create a dataframe with customer id and last purchase date history before
tx_last_purchase = df_before.groupby('customer_id').date_order_day.max().reset_index()
tx_last_purchase.columns = ['customer_id','maxpurchaseday']
tx_purchase_dates = pd.merge(tx_last_purchase, tx_next_first_purchase, on='customer_id', how='left')
tx_purchase_dates['next_pur_day'] = (tx_purchase_dates['minpurchaseday'] - tx_purchase_dates['maxpurchaseday']).dt.days
print(tx_purchase_dates)


# In[151]:


tx_purchase_dates = tx_purchase_dates[tx_purchase_dates.next_pur_day.notnull()]


# In[152]:


#plot next purchase day
plot_data = [
    go.Histogram(
        x=tx_purchase_dates.query('next_pur_day > 0')['next_pur_day']
    )
]

plot_layout = go.Layout(
        title='Next purchase day'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


# In[153]:


#merge with tx_user_before
tx_user_before = pd.merge(tx_user_before, tx_purchase_dates[['customer_id','next_pur_day']],on='customer_id',how='left')


# In[154]:


have_label = tx_user_before[tx_user_before.next_pur_day.notnull()]


# # GROUP 3

# In[155]:


extract_time = data_h_filter.copy()


# In[156]:


extract_time['year'] = pd.DatetimeIndex(extract_time['date_order_day']).year
extract_time['week'] = pd.DatetimeIndex(extract_time['date_order_day']).week
extract_time['co_time'] = (extract_time['year'] -  extract_time['year'].min())*52+extract_time['week']
df_time = extract_time[['customer_id','co_time']]


# In[157]:


df_time = df_time.drop_duplicates(subset=['customer_id','co_time'], keep='first')
# shift row order day
df_time['shift_order'] = (df_time.sort_values(by=['co_time'], 
                ascending = False).groupby(['customer_id'])['co_time'].shift(-1))
# sort_values từ last order day
df_time = df_time.sort_values(by='co_time', ascending = False)
df_time['timediff'] = df_time['co_time'] - df_time['shift_order']


# In[158]:


cal_std_time = df_time.groupby('customer_id').timediff.std().reset_index()
cal_std_time.columns=['customer_id','std_time']


# In[159]:


df_time_uni = df_time.groupby('customer_id').co_time.nunique().reset_index()
df_time_uni.columns=['customer_id','uni_time']


# In[160]:


df_time_merge = pd.merge(df_time_uni, cal_std_time, on='customer_id', how='left')


# In[161]:


filter_month_frequency=df_time_merge[(df_time_merge.std_time==0)&(df_time_merge.uni_time>=4)&(df_time_merge.customer_id.isin(have_label.customer_id))]
print('GROUP 3: Customer buy every Week: ',filter_month_frequency.customer_id.count())


# In[162]:


df_before = df_before[df_before.customer_id.isin(have_label.customer_id)]
df_after = df_after[df_after.customer_id.isin(have_label.customer_id)]


# # GROUP 3: 42
extract_month[extract_month.customer_id==437469].sort_values(by='date_order_day', ascending=True)# Remove customer have month frequency
df_before = df_before[~df_before.customer_id.isin(filter_month_frequency.customer_id)]
df_after = df_after[~df_after.customer_id.isin(filter_month_frequency.customer_id)]
# In[163]:


print('Customer have lable: ',df_before.customer_id.nunique())


# # PREPROCESSING, GENERATE FEATURES

# # 2.1 Feature revenue by co-time

# In[164]:


extract_time = df_before.copy()


# In[165]:


extract_time['year'] = pd.DatetimeIndex(extract_time['date_order_day']).year
extract_time['day_of_year'] = extract_time['date_order_day'].dt.dayofyear
extract_time['day_of_year'] = (extract_time['year'] -  extract_time['year'].min())*(365+floor((floor(extract_time['year'].iloc[0]/4) - (extract_time['year'].iloc[0]/4))+1))+extract_time['day_of_year']


# In[166]:


time_step=5
time_list=extract_time.sort_values('day_of_year')['day_of_year'].unique()[::time_step]
def split_time_step(extract_time):
    global val1
    for i in time_list:
        if extract_time.day_of_year >= i:
            val1 = np.where(time_list==i)[0][0]
    return val1
extract_time['co_time'] = extract_time.apply(split_time_step, axis=1)
extract_time['co_time']=extract_time['co_time']+1


# In[167]:


df_m = extract_time[['customer_id','co_time','amount']]


# In[168]:


m_score=df_m.groupby(['customer_id','co_time']).amount.sum().reset_index()
m_score.columns=['customer_id','m','revenue']


# In[169]:


#Revenue clusters 
kmeans_revenue = KMeans(n_clusters=4, random_state=3)
kmeans_revenue.fit(m_score[['revenue']])
m_score['revenue_cluster'] = kmeans_revenue.predict(m_score[['revenue']])


# In[170]:


import pickle
with open('D:\Python\Data\Model/kmean_revenue_hlc_now_ver1_20200303.pkl', 'wb') as file_saved:
        pickle.dump(kmeans_revenue, file_saved)


# In[171]:


CL_revenue=m_score.groupby('revenue_cluster')['revenue'].describe().sort_values('min').reset_index().reset_index()
CL_revenue


# In[172]:


CL_revenue['index'] = CL_revenue['index']+1
CL_revenue = CL_revenue[['index','revenue_cluster']]
CL_revenue = CL_revenue.rename(columns={'index': 'm_score'})
m_score = pd.merge(m_score, CL_revenue, on='revenue_cluster', how='left')
m_score = m_score.drop(columns=['revenue_cluster'])


# In[173]:


m_pivot = m_score.pivot(index='customer_id', columns='m', values='m_score').reset_index()
m_pivot = m_pivot.fillna(0)
m_pivot=m_pivot.set_index('customer_id')
m_pivot=m_pivot.stack().reset_index()
m_pivot.columns=['customer_id','m','m_score']


# In[174]:


print('FEATURE 2.1 REVENUE')
print(m_pivot)


# # 2.2 Feature Frequency by co-time

# In[175]:


df_f = extract_time[['customer_id','co_time','date_order_day']]


# In[176]:


f_score=df_f.groupby(['customer_id','co_time']).date_order_day.nunique().reset_index()
f_score.columns=['customer_id','m','frequency']


# In[177]:


#Frequency clusters 
kmeans_frequency = KMeans(n_clusters=3, random_state=3)
kmeans_frequency.fit(f_score[['frequency']])
f_score['frequency_cluster'] = kmeans_frequency.predict(f_score[['frequency']])


# In[178]:


import pickle
with open('D:\Python\Data\Model/kmean_frequency_hlc_now_update_03_03.pkl', 'wb') as file_saved:
        pickle.dump(kmeans_frequency, file_saved)


# In[179]:


CL_frequency=f_score.groupby('frequency_cluster')['frequency'].describe().sort_values('min').reset_index().reset_index()
CL_frequency


# In[180]:


CL_frequency['index'] = CL_frequency['index']+1
CL_frequency = CL_frequency[['index','frequency_cluster']]
CL_frequency = CL_frequency.rename(columns={'index': 'f_score'})
f_score = pd.merge(f_score, CL_frequency, on='frequency_cluster', how='left')
f_score = f_score.drop(columns=['frequency_cluster'])


# In[181]:


f_pivot = f_score.pivot(index='customer_id', columns='m', values='f_score').reset_index()
f_pivot = f_pivot.fillna(0)
f_pivot=f_pivot.set_index('customer_id')
f_pivot=f_pivot.stack().reset_index()
f_pivot.columns=['customer_id','m','f_score']


# In[182]:


print('FEATURE 2.2 Frequency')
print(f_pivot)


# # 2.4 Feature Quantity

# In[183]:


df_qty = extract_time[extract_time.amount>0][['customer_id','co_time','qty']]


# In[184]:


qty_score=df_qty.groupby(['customer_id','co_time']).qty.sum().reset_index()
qty_score.columns=['customer_id','m','qty']


# In[185]:


#Quantity clusters 
kmeans_qty = KMeans(n_clusters=2, random_state=3)
kmeans_qty.fit(qty_score[['qty']])
qty_score['qty_cluster'] = kmeans_qty.predict(qty_score[['qty']])


# In[186]:


import pickle
with open('D:\Python\Data\Model/kmean_quantity_hlc_now_ver1_update_03_03.pkl', 'wb') as file_saved:
        pickle.dump(kmeans_qty, file_saved)


# In[187]:


CL_quantity=qty_score.groupby('qty_cluster')['qty'].describe().sort_values('min').reset_index().reset_index()
CL_quantity


# In[188]:


CL_quantity= qty_score.groupby('qty_cluster')['qty'].describe().sort_values('min').reset_index().reset_index()
CL_quantity['index'] = CL_quantity['index']+1
CL_quantity = CL_quantity[['index','qty_cluster']]
CL_quantity = CL_quantity.rename(columns={'index': 'qty_score'})
qty_score = pd.merge(qty_score, CL_quantity, on='qty_cluster', how='left')
qty_score = qty_score.drop(columns=['qty_cluster'])


# In[189]:


qty_pivot = qty_score.pivot(index='customer_id', columns='m', values='qty_score').reset_index()
qty_pivot = qty_pivot.fillna(0)
qty_pivot = qty_pivot.set_index('customer_id')
qty_pivot = qty_pivot.stack().reset_index()
qty_pivot.columns=['customer_id','m','qty_score']


# In[190]:


print(qty_pivot)


# # 2.4 Covid-19

# In[191]:


def lstm_preprocessing(df_raw, col_f: str):
    df_raw = df_raw.drop_duplicates(subset=['customer_id','co_time'])
    fea_pivot = df_raw.pivot(index = "customer_id", columns = 'co_time', values = col_f).reset_index()
    fea_pivot = fea_pivot.fillna(0)
    fea_pivot = fea_pivot.set_index('customer_id')
    fea_pivot = fea_pivot.stack().reset_index()
    fea_pivot.columns=['customer_id', 'm', col_f]
    return fea_pivot


# In[192]:


# df_cov = extract_time[extract_time.qty>0][['customer_id','co_time','confirmed']]
df_cov = extract_time[['customer_id','co_time','confirmed']]
df_cov_confirm = lstm_preprocessing(df_cov, 'confirmed')


# In[193]:


df_cov = extract_time[['customer_id','co_time','active']]
df_cov_active = lstm_preprocessing(df_cov, 'active')


# In[194]:


df_cov = extract_time[['customer_id','co_time','recovered']]
df_cov_recover = lstm_preprocessing(df_cov, 'recovered')


# # Step 3: Merge every data frame together
m_pivot
f_pivot
qty_pivot
cov_pivot
# In[195]:


features = pd.merge(m_pivot, f_pivot, on=['customer_id','m'], how='right')


# In[196]:


features = pd.merge(features, qty_pivot, on=['customer_id','m'], how='left')


# In[197]:


features = pd.merge(features, df_cov_confirm, on=['customer_id','m'], how='left')
features = pd.merge(features, df_cov_active, on=['customer_id','m'], how='left')
features = pd.merge(features, df_cov_recover, on=['customer_id','m'], how='left')


# In[198]:


features.info()


# In[199]:


features.max()


# In[200]:


features = features.fillna(0)


# In[201]:


features = features.sort_values(['customer_id','m'])


# In[202]:


t=features.m.nunique()


# In[203]:


features = features.set_index('customer_id')
features = features.drop(columns=['m'])


# In[204]:


features


# In[205]:


label = pd.merge(features, tx_purchase_dates, on='customer_id', how='left')
label = label[['customer_id','next_pur_day']]
label = label.sort_values('customer_id')
label = label.drop_duplicates(subset='customer_id')
label = label.set_index('customer_id')


# In[206]:


X=features.values.copy()
y=label.values.copy()


# In[207]:


X=X.reshape(-1,t,len(features.columns))
print('X')
print(X[0])
print('y')
print(y)


# In[208]:


#split data train & test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[209]:


print('X_train shape')
print(X_train.shape)
print(X_train)


# In[210]:


model = Sequential()
model.add(LSTM(40, batch_size=20,#stateful=True, batch_input_shape=(1, X_train.shape[1], X_train.shape[2]), 
               input_shape=(X_train.shape[1], X_train.shape[2]), recurrent_dropout=0.1))
#model.add(Embedding(1000, 64, input_length=128))
model.add(Dense(90, activation='relu'))
model.add(Dense(1))

model.compile(optimizer=tf.keras.optimizers.Adam(#learning_rate=0.0001
                                                   ),
              loss='mae',
              metrics=['mae', 'mse'])
model.summary()


# In[211]:


callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.02, patience=50, verbose=0, mode='auto', baseline=None, restore_best_weights=True)
#callback = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)


# In[212]:


history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test),
                                                    callbacks=[callback])


# In[213]:


hist = pd.DataFrame(history.history)
a=hist.val_mae.min()
b=hist[hist.val_mae==a].reset_index()
c=b['index'].iloc[0]
print('Epochs min val_mae:')
print(hist[hist.val_mae==a].reset_index().iloc[0,0])
plotter = tfdocs.plots.HistoryPlotter(smoothing_std=0.1)
plotter.plot({'Basic': history}, metric = "mae")
plt.ylim([a-1, a+2])
plt.xlim([c-60, c+50])
plt.ylabel('MAE [MPG]')

# Recreate the exact same model, including its weights and the optimizer
model = tf.keras.models.load_model('model_next_purchase_hlc_now_Adam_regression_best_v1.h5')
# In[214]:


y_pred = model.predict(X_test)


# In[215]:


y_test = pd.DataFrame(y_test)
y_test.columns=['next_pur_day']


# In[216]:


y_pred=pd.DataFrame(y_pred)
y_pred.columns=['y_pred']


# In[217]:


check = pd.merge(y_test, y_pred, left_index=True, right_index=True, how='left')


# In[218]:


check['y_error'] = (check['next_pur_day'] - y_pred['y_pred']).abs()
check['y_error_percent'] = check['y_error']/check['next_pur_day']


# In[219]:


# check next_pur_range
check.loc[check.next_pur_day >= 0, 'next_pur_range'] = 5
check.loc[check.next_pur_day > 15,'next_pur_range'] = 4
check.loc[check.next_pur_day > 30,'next_pur_range'] = 3
check.loc[check.next_pur_day > 60,'next_pur_range'] = 2
check.loc[check.next_pur_day > 90,'next_pur_range'] = 1


# In[220]:


# check predict range
check.loc[check.y_pred >= 0, 'pred_range'] = 5
check.loc[check.y_pred > 15,'pred_range'] = 4
check.loc[check.y_pred > 30,'pred_range'] = 3
check.loc[check.y_pred > 60,'pred_range'] = 2
check.loc[check.y_pred > 90,'pred_range'] = 1


# In[221]:


wing=5
def f1score(check):
    if check['next_pur_range'] <= 1:
        if check['y_pred'] > 90:
            val=1
        else:
            val=0
    elif check['next_pur_range'] > 1:
        if check['y_error'] <= wing:
            val=1
        else:
            val=0
    return val
check['f1_score'] = check.apply(f1score, axis=1)


# In[222]:


check_sort = check.sort_values('next_pur_day').reset_index().reset_index()
check_sort['y_pred'] =check_sort['y_pred'].round()
check_sort['plus10']=check_sort['next_pur_day']+wing
check_sort['minus10']=check_sort['next_pur_day']-wing


# In[223]:


def right_func(check_sort):
    if check_sort['y_pred'] <= check_sort['plus10']:
        if check_sort['y_pred'] >= check_sort['minus10']:
            val=1
        else:
            val=0
    elif check_sort['y_pred'] > check_sort['plus10']:
        val=0
    return val
check_sort['f1_score'] = check_sort.apply(right_func, axis=1)


# In[224]:


import plotly.express as px
fig = px.line(check_sort, x="level_0", y="next_pur_day", color="next_pur_range",  height=600)
fig.add_scatter(x=check_sort[check_sort.f1_score==1]['level_0'], y=check_sort[check_sort.f1_score==1]['y_pred'], mode='markers', name='Pred Right')
fig.add_scatter(x=check_sort[check_sort.f1_score==0]['level_0'], y=check_sort[check_sort.f1_score==0]['y_pred'], mode='markers', name='Pred Wrong')
fig.add_scatter(x=check_sort['level_0'], y=check_sort['plus10'], mode='lines', name='range +'+str(wing))
fig.add_scatter(x=check_sort['level_0'], y=check_sort['minus10'], mode='lines', name='range -'+str(wing))
fig.show()
fig.write_html("./have_covid.html")


# In[225]:


f1_results = check.groupby('next_pur_range').f1_score.sum()
f1_results = check.groupby('next_pur_range').f1_score.sum().reset_index()
f1_results.columns=['next_pur_range','f1_score']


# In[226]:


support = check.groupby('next_pur_range')['next_pur_day'].count().reset_index()
support.columns=['next_pur_range','support']


# In[227]:


results = pd.merge(f1_results, support, on='next_pur_range')


# In[228]:


results['f1_score_percent'] = results['f1_score']/results['support']


# In[229]:


from sklearn.metrics import confusion_matrix
print(results)
acc=check.f1_score.sum()
total=check.next_pur_day.count()
print('                               Accuracy avg: ' ,acc/total)
print(pd.DataFrame(confusion_matrix(check.next_pur_range, check.pred_range)))


# In[230]:


check['pred_range'] = check['pred_range'].fillna(4)


# In[231]:


model.summary()

# end and save model
model.save('model_next_purchase_hlc_now_Adam_regression_best_covid_v1.h5')