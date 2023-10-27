#!/usr/bin/env python
# coding: utf-8

# In[1]:


############################################
########## My common Libraries #############
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
from matplotlib import dates
import seaborn as sns
import datetime as datetime
#import scipy as sp
import statsmodels.api as sm
import sklearn
import random as r
###########################################
###########################################

############################################
##########  ignore warnings ################
import warnings
warnings.filterwarnings('ignore')
###########################################
############################################


# In[ ]:





# #  Steps
# 
# ### 1.) Abstract
# 
# 
# 
# ### 2.) Introduction
# 
# [2.1] Intro
# 
# [2.2] Methodology
# 
# 
# ---
# 
# 
# ### 3.) Exploratory Data Analysis 
# 
# [3.1] Variable Identification 
# 
# [3.2] Univariate Analysis 
# 
# [3.3] Missing Data 
# 
# [3.4] Initial Observations 
# 
# ---
# 
# ### 4.) Feature Engineering
# 
# [4.1] Handling Missing Data  
# 
# [4.2] Variable Scaling
# 
# ---
# 
# ### 5.) Time Series Analysis
# 
# [5.1] Checking Assumptions
# 
# [A] Checking For Stationarity 
# 
# [B] Checking For Seasonality
# 
# [5.2] Transformations(Differencing)
# 
# [5.3] Train Test Split
# 
# [5.4] Model Building
# 
# [5.5] Model Diagnostics(Residual Analysis)
# 
# [5.6] Detecting Outliers
# 
# [5.7] Model Selection
# 
# [5.8] Forecasting
# 
# ---
# 
# ---
# 
# ### 6.) Conclusion
# 
# ---
# 
# ### 7.) Weaknesses 
# 
# ---
# 
# 

# # $\color{red}{\textbf{1 Abstract}}$

# In[ ]:





# # $\color{red}{\textbf{2 Introduction}}$

# [2.1] (Intro) This step is done(check document)
# 
# [2.2] (Methodology) This step is done

# # $\color{red}{\textbf{3 Exploratory Data Analysis}}$

# In[2]:


#### Now eda begins ###

### upload data ###
EurUsd_data = pd.read_csv('EURUSD_daily.csv') # 00:00 --> 4:00 pm est --> 16:00:00
EurUsd_data = EurUsd_data.iloc[:-1 , :] # select all rows except last one

EurUsd_data.head()


# In[3]:


EurUsd_data.tail()


# In[8]:


### skip
### create a dictionary to convert Time(hr) to est ###
est_time = {'0:00':'16:00:00','1:00':'17:00:00','2:00':'18:00:00','3:00':'19:00:00','4:00':'20:00:00',
            '5:00':'21:00:00','6:00':'22:00:00','7:00':'23:00:00','8:00':'00:00:00','9:00':'1:00:00',
            '10:00':'2:00:00','11:00':'3:00:00','12:00':'4:00:00','13:00':'5:00:00','14:00':'6:00:00',
            '15:00':'7:00:00','16:00':'8:00:00','17:00':'9:00:00','18:00':'10:00:00','19:00':'11:00:00',
            '20:00':'12:00:00','21:00':'13:00:00','22:00':'14:00:00','23:00':'15:00:00'}
 
    
### Time(hrs) est miltary time ###
EurUsd_data['Time(hr)_Est'] = EurUsd_data['Time(hr)'].map(est_time)
EurUsd_data.head()


# In[ ]:


### skip
### combine Date with est time ###
EurUsd_data['Date'] = EurUsd_data.apply(lambda x: x['Date'] +' '+x['Time(hr)_Est'], axis=1 )
EurUsd_data.head()


# In[4]:


### convert the Date into a datetime object ###
EurUsd_data['Date']  = pd.to_datetime( EurUsd_data['Date'] )

### make a week day column ###
EurUsd_data['Day'] = EurUsd_data['Date'].dt.weekday
EurUsd_data.head()


# In[5]:


### make the Date column my index
EurUsd_data.set_index('Date', inplace=True)
EurUsd_data.head()


# In[6]:


EurUsd_data.tail()


# In[7]:


### checking index ###
print( EurUsd_data.index ) ### my freq is none, which is bad


# In[8]:


len(EurUsd_data)


# In[9]:


### resample data to have business days only ###
EurUsd_data = EurUsd_data.resample(rule='B').mean()
EurUsd_data.head()


# In[10]:


### checking index ###
print( EurUsd_data.index ) ### my freq is now business days


# In[11]:


len(EurUsd_data)
### set freq index to business days ###
#len(EurUsd_data.asfreq('b').index)


# In[12]:


print( EurUsd_data.index ) ### my freq is none, which is bad


# In[13]:


EurUsd_data['Day'].unique()


# In[ ]:





# In[14]:


EurUsd_data.tail()


# In[ ]:





# ### Columns

# In[14]:


EurUsd_data.columns


# In[16]:


# EurUsd_close = EurUsd_data['Close'].copy()


# In[ ]:





# In[ ]:





# ### Variable Analysis

# In[ ]:





# ### The type of variables in the dataset

# In[ ]:





# ### statisics analysis

# In[15]:


# overall description
EurUsd_data.describe()


# In[18]:


# description in 2020
EurUsd_data['2020':'2020'].describe()


# In[19]:


# description in october 2021
EurUsd_data['2021-10':'2021-10'].describe()


# ### graph analysis

# In[20]:


fig, axes = plt.subplots(figsize = (6,2),dpi=200)
style.use('seaborn-whitegrid')

EurUsd_data['High'].plot(legend=True,color='red')
EurUsd_data['Low'].plot(legend=True,color='blue')

axes.set_title('EUR/USD Exchange Rate from 2012 to 2021')
axes.autoscale(axis = 'x',tight=True)
axes.set_ylabel('Exchange Rate');


# In[74]:


fig, axes = plt.subplots(figsize = (6,2),dpi=200)
style.use('seaborn-whitegrid')

EurUsd_data['High']['2021':].plot(legend=True,color='red')
EurUsd_data['Low']['2021':].plot(legend=True,color='blue')

axes.set_title('EUR/USD Exchange Rate Over 2021')
axes.autoscale(axis = 'x',tight=True)
axes.set_ylabel('Exchange Rate');


# In[22]:


fig, axes = plt.subplots(figsize = (12,6),dpi=200)
style.use('seaborn-whitegrid')

EurUsd_data['High']['2021-10':'2021-10'].plot(legend=True,color='red')
EurUsd_data['Low']['2021-10':'2021-10'].plot(legend=True,color='blue')

axes.set_title('EUR/USD Exchange Rate Over Time In October 2021')
axes.autoscale(axis = 'x',tight=True)
axes.set_ylabel('Exchange Rate');


# In[ ]:





# In[68]:


fig, axes = plt.subplots(figsize = (12,6),dpi=200)
style.use('seaborn-whitegrid')

axes.set_title('EUR/USD Exchange Rate In October 2021')

dd=pd.melt(EurUsd_data.copy()['2021-10':'2021-10'],
           id_vars=['Day'],
           value_vars=['High','Low'],
           var_name='Level')

sns.boxplot(x='Day',y='value' , data=dd,hue='Level',palette='rocket');
axes.set_ylabel('Exchange Rate')
axes.set_xticklabels(['Monday',"Tuesday","Wednesday","Thursday","Friday"]);


# In[69]:


fig, axes = plt.subplots(figsize = (12,6),dpi=200)
style.use('seaborn-whitegrid')

axes.set_title('EUR/USD Exchange Rate In September 2021')

dd=pd.melt(EurUsd_data.copy()['2021-09':'2021-09'],
           id_vars=['Day'],
           value_vars=['High','Low'],
           var_name='Level')

sns.boxplot(x='Day',y='value' , data=dd,hue='Level',palette='rocket');
axes.set_ylabel('Exchange Rate')
axes.set_xticklabels(['Monday',"Tuesday","Wednesday","Thursday","Friday"]);


# In[70]:


fig, axes = plt.subplots(figsize = (12,6),dpi=200)
style.use('seaborn-whitegrid')

axes.set_title('EUR/USD Exchange Rate In August 2021')

dd=pd.melt(EurUsd_data.copy()['2021-08-01':'2021-08-30'],
           id_vars=['Day'],
           value_vars=['High','Low'],
           var_name='Level')

sns.boxplot(x='Day',y='value' , data=dd,hue='Level',palette='rocket');
axes.set_ylabel('Exchange Rate')
axes.set_xticklabels(['Monday',"Tuesday","Wednesday","Thursday","Friday"]);


# In[71]:


fig, axes = plt.subplots(figsize = (12,6),dpi=200)
style.use('seaborn-whitegrid')

axes.set_title('EUR/USD Exchange Rate In July 2021')

dd=pd.melt(EurUsd_data.copy()['2021-07':'2021-07'],
           id_vars=['Day'],
           value_vars=['High','Low'],
           var_name='Level')

sns.boxplot(x='Day',y='value' , data=dd,hue='Level',palette='rocket');
axes.set_ylabel('Exchange Rate')
axes.set_xticklabels(['Monday',"Tuesday","Wednesday","Thursday","Friday"]);


# In[72]:


fig, axes = plt.subplots(figsize = (12,6),dpi=200)
style.use('seaborn-whitegrid')

axes.set_title('EUR/USD Exchange Rate In June 2021')

dd=pd.melt(EurUsd_data.copy()['2021-06':'2021-06'],
           id_vars=['Day'],
           value_vars=['High','Low'],
           var_name='Level')

sns.boxplot(x='Day',y='value' , data=dd,hue='Level',palette='rocket');
axes.set_ylabel('Exchange Rate')
axes.set_xticklabels(['Monday',"Tuesday","Wednesday","Thursday","Friday"]);


# In[73]:


fig, axes = plt.subplots(figsize = (12,6),dpi=200)
style.use('seaborn-whitegrid')

axes.set_title('EUR/USD Exchange Rate In May 2021')

dd=pd.melt(EurUsd_data.copy()['2021-05':'2021-05'],
           id_vars=['Day'],
           value_vars=['High','Low'],
           var_name='Level')

sns.boxplot(x='Day',y='value' , data=dd,hue='Level',palette='rocket');
axes.set_ylabel('Exchange Rate')
axes.set_xticklabels(['Monday',"Tuesday","Wednesday","Thursday","Friday"]);


# ### Decompose analysis

# In[23]:


# 2021 conponents high

from statsmodels.tsa.seasonal import seasonal_decompose
plt.rc("figure", figsize=(8,4),dpi=200)

result = seasonal_decompose(x=EurUsd_data['High']['2021':'2021'].dropna());
result.plot();


# In[24]:


# 2021 conponents Low
from statsmodels.tsa.seasonal import seasonal_decompose
plt.rc("figure", figsize=(8,4),dpi=200)
result = seasonal_decompose(x=EurUsd_data['Low']['2021':'2021'].dropna());
result.plot();


# In[25]:


# 2021 October conponents high
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(x=EurUsd_data['High']['2021-10':'2021-10'].dropna());
result.plot();


# In[26]:


# 2021 October conponents high
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(x=EurUsd_data['Low']['2021-10':'2021-10'].dropna());
result.plot();


# In[ ]:





# In[ ]:





# # $\color{red}{\textbf{4 Variable Engineering}}$

# ### missing data 

# In[16]:


EurUsd_data.isnull().sum()


# ### Day missing col

# In[17]:


EurUsd_data['Day'] = EurUsd_data.index.weekday
EurUsd_data.isnull().sum()


# In[18]:


### show the rows of nan values
EurUsd_data[EurUsd_data.isna().any(axis=1)]


# In[19]:


### we will use last observation carried forward method ###
EurUsd_data = EurUsd_data.ffill(axis = 0)


# In[20]:


EurUsd_data.isnull().sum()


# In[21]:


EurUsd_data.head()


# In[ ]:





# In[ ]:





# In[ ]:





# ### Variable Scaling

# In[22]:


EurUsd_data_high = EurUsd_data[['High','Day']].copy()
EurUsd_data_low = EurUsd_data[['Low','Day']].copy()


# In[ ]:





# In[23]:


### log scaling ###
log_transformer = lambda x: np.log(x)
invert_log_transformer = lambda x: np.e**(x)

EurUsd_data_high['High_log'] = log_transformer(EurUsd_data_high['High'])
EurUsd_data_low['Low_log'] = log_transformer(EurUsd_data_low['Low'])


# In[24]:


### recipical scaling ###
recipical_transformer = lambda x: 1/x
invert_recipical_transformer = lambda x: x**(-1) 

EurUsd_data_high['High_recip'] = recipical_transformer(EurUsd_data_high['High'])
EurUsd_data_low['Low_recip'] = recipical_transformer(EurUsd_data_low['Low'])


# In[25]:


### sqroot scaling ###
sqrt_transformer = lambda x: x**(1/2)
invert_sqrt_transformer = lambda x: x**(2) 

EurUsd_data_high['High_sqrt'] = sqrt_transformer(EurUsd_data_high['High'])
EurUsd_data_low['Low_sqrt'] = sqrt_transformer(EurUsd_data_low['Low'])


# In[26]:


### sq scaling ###
sq_transformer = lambda x: x**(2)
invert_sq_transformer = lambda x: x**(1/2) 

EurUsd_data_high['High_sq'] = sq_transformer(EurUsd_data_high['High'])
EurUsd_data_low['Low_sq'] = sq_transformer(EurUsd_data_low['Low'])


# In[27]:


### box-cox scaling ###
from scipy.stats import boxcox 

boxcox_transformer = lambda x: boxcox(x)[0]

EurUsd_data_high['High_box'] = boxcox_transformer(EurUsd_data_high['High'])
EurUsd_data_low['Low_box'] = boxcox_transformer(EurUsd_data_low['Low'])

####################################################################################
from scipy.special import inv_boxcox
lambda_high = boxcox(EurUsd_data_high['High'])[1]
lambda_low = boxcox(EurUsd_data_low['Low'])[1]

invert_boxcox_transformer_high = lambda x,y: inv_boxcox(x,lambda_high) 
invert_boxcox_transformer_low = lambda x,y: inv_boxcox(x,lambda_low) 


# In[28]:


EurUsd_data_high.columns


# In[29]:


EurUsd_data_low.columns


# In[30]:


fig, axes = plt.subplots(figsize = (10,5),dpi=200)
style.use('seaborn-whitegrid')

EurUsd_data_high[['High', 'High_log', 'High_recip', 'High_sqrt', 'High_sq', 'High_box']].plot(legend=True,ax=axes);


axes.set_title('EUR/USD Exchange Rate from 2012 to 2021 High');
axes.autoscale(axis = 'x',tight=True);
axes.set_ylabel('Exchange Rate');


# In[31]:


fig, axes = plt.subplots(figsize = (10,5),dpi=200)
style.use('seaborn-whitegrid')

EurUsd_data_low[['Low', 'Low_log', 'Low_recip', 'Low_sqrt', 'Low_sq', 'Low_box']].plot(legend=True,ax=axes);


axes.set_title('EUR/USD Exchange Rate from 2012 to 2021 Low');
axes.autoscale(axis = 'x',tight=True);
axes.set_ylabel('Exchange Rate');


# In[ ]:





# In[ ]:





# # $\color{red}{\textbf{5 Time Series Analysis}}$

# [5.1] Checking Assumptions
# 
# [A] Checking For Stationarity
# 
# [B] Checking For Seasonality
# 
# [5.2] Transformations(Differencing)
# 
# [5.3] Train Test Split
# 
# [5.4] Model Building
# 
# [5.5] Model Diagnostics(Residual Analysis)
# 
# [5.6] Detecting Outliers
# 
# [5.7] Model Selection
# 
# [5.8] Forecasting

# ## Checking Assumptions

# ### Checking For Stationarity

# In[32]:


from statsmodels.tsa.stattools import adfuller


# In[33]:


### make a list of col names high
EurUsd_data_high_cols = list(EurUsd_data_high.columns)
del EurUsd_data_high_cols[1]


# In[34]:


### make a list of col names low
EurUsd_data_low_cols = list(EurUsd_data_low.columns)
del EurUsd_data_low_cols[1]


# In[35]:


def dickey_fuller_test(x,p_value):
    result = adfuller(x)
    print("Test-Statistic:", result[0])
    print("P-Value:", result[1])
    if result[1] < p_value:
        print("We reject the null hypothesis.")
        print("There is strong evidence that the time series has no unit root and is stationary.")
    else:
        print("We failed to reject the null hypothesis.")
        print("There is little evidence that the time series has a unit root and is not stationary.")


# In[36]:


### High stationary ###
for i in EurUsd_data_high_cols:
    print(i)
    dickey_fuller_test(EurUsd_data_high[i],0.05)
    print('###########\n')


# In[37]:


### Low stationary ###
for i in EurUsd_data_low_cols:
    print(i)
    dickey_fuller_test(EurUsd_data_low[i],0.05)
    print('###########\n')


# ### Differencing

# In[38]:


from statsmodels.tsa.statespace.tools import diff

difference_transformer = lambda x: diff(x,k_diff=1)


# In[39]:


### Difference 1 ###
for i in EurUsd_data_high_cols:
    EurUsd_data_high[i+'_DIFF1'] = difference_transformer(EurUsd_data_high[i])
    
for i in EurUsd_data_low_cols:
    EurUsd_data_low[i+'_DIFF1'] = difference_transformer(EurUsd_data_low[i])


# In[40]:


### High stationary ###
for i in EurUsd_data_high_cols:
    print(i)
    dickey_fuller_test(EurUsd_data_high[i],0.05)
    print('###########\n')


# In[41]:


### High stationary ###
for i in list(EurUsd_data_high.columns)[-6:]:
    print(i)
    dickey_fuller_test(EurUsd_data_high[i].dropna(),0.05)
    print('###########\n')


# In[42]:


### Low stationary ###
for i in list(EurUsd_data_low.columns)[-6:]:
    print(i)
    dickey_fuller_test(EurUsd_data_low[i].dropna(),0.05)
    print('###########\n')


# In[43]:


fig, axes = plt.subplots(figsize = (16,10),dpi=200)
style.use('seaborn-whitegrid')

EurUsd_data_high[['High_DIFF1', 'High_log_DIFF1', 'High_recip_DIFF1', 'High_sqrt_DIFF1', 'High_sq_DIFF1', 'High_box_DIFF1']].plot(legend=True,ax=axes);


axes.set_title('EUR/USD Exchange Rate from 2012 to 2021 High');
axes.autoscale(axis = 'x',tight=True);
axes.set_ylabel('Exchange Rate');


# In[44]:


EurUsd_data_high.head()


# In[45]:


fig, axes = plt.subplots(figsize = (16,10),dpi=200)
style.use('seaborn-whitegrid')

EurUsd_data_low[['Low_DIFF1', 'Low_log_DIFF1', 'Low_recip_DIFF1', 'Low_sqrt_DIFF1', 'Low_sq_DIFF1', 'Low_box_DIFF1']].plot(legend=True,ax=axes);


axes.set_title('EUR/USD Exchange Rate from 2012 to 2021 Low');
axes.autoscale(axis = 'x',tight=True);
axes.set_ylabel('Exchange Rate');


# ### Train Test validate Split

# In[ ]:





# In[ ]:


### skip ###
#### test set last 90 days #########################
test_high_set = EurUsd_data_high[-90:].copy() #
test_low_set = EurUsd_data_low[-90:].copy() #
####################################################
####################################################
####################################################

#### training and validation set set last 90 days ##
high_set = EurUsd_data_high[:-90].copy()  
low_set = EurUsd_data_low[:-90].copy()

train_high_80 = int(len(high_set)*0.80//1)
train_low_80 = int(len(low_set)*0.80//1)

train_high_set = high_set[:train_high_80].copy() #
validation_high_set = high_set[train_high_80:].copy() #

train_low_set = low_set[:train_low_80].copy() #
validation_low_set = low_set[train_low_80:].copy() #
####################################################
####################################################
####################################################


# In[562]:


#### test set last 90 days #########################
test_high_set = EurUsd_data_high[-90:].copy() #
test_low_set = EurUsd_data_low[-90:].copy() #
####################################################
####################################################
####################################################

#### training and validation set set last 90 days ##
high_set = EurUsd_data_high[:-90].copy()  
low_set = EurUsd_data_low[:-90].copy()

train_high_80 = int(len(high_set)*0.80//1)
train_low_80 = int(len(low_set)*0.80//1)
################################################################################################################

### TRAINING SET AND VALIDATION SET ###
from pmdarima import model_selection

train_high_set, validation_high_set = model_selection.train_test_split(high_set, train_size = train_high_80)
train_low_set, validation_low_set = model_selection.train_test_split(low_set, train_size = train_low_80)
################################################################################################################


# In[ ]:





# In[ ]:





# In[ ]:





# ##  Model Building

# ## ACF and PACF graphs

# In[47]:


from statsmodels.graphics.tsaplots import plot_pacf, plot_acf


# In[58]:


### plot acf high ###

fig, axes = plt.subplots(figsize=(10, 3))
plot_acf(train_high_set['High_DIFF1'].dropna(), ax=axes,lags=50);


### plot pacf high ###

fig, axes = plt.subplots(figsize=(10, 3))
plot_pacf(train_high_set['High_DIFF1'].dropna(), ax=axes,lags=50);


# In[114]:


### plot acf high ###

fig, axes = plt.subplots(figsize=(10, 5))
plot_acf(train_high_set['High_log_DIFF1'].dropna(), ax=axes,lags=50);

### plot pacf high ###

fig, axes = plt.subplots(figsize=(10, 5))
plot_pacf(train_high_set['High_log_DIFF1'].dropna(), ax=axes,lags=50);


# In[115]:


### plot acf high ###

fig, axes = plt.subplots(figsize=(10, 5))
plot_acf(train_high_set['High_recip_DIFF1'].dropna(), ax=axes,lags=50);

### plot pacf high ###

fig, axes = plt.subplots(figsize=(10, 5))
plot_pacf(train_high_set['High_recip_DIFF1'].dropna(), ax=axes,lags=50);


# In[116]:


### plot acf high ###

fig, axes = plt.subplots(figsize=(10, 5))
plot_acf(train_high_set['High_sqrt_DIFF1'].dropna(), ax=axes,lags=50);

### plot pacf high ###

fig, axes = plt.subplots(figsize=(10, 5))
plot_pacf(train_high_set['High_sqrt_DIFF1'].dropna(), ax=axes,lags=50);


# In[117]:


### plot acf high ###

fig, axes = plt.subplots(figsize=(10, 5))
plot_acf(train_high_set['High_sq_DIFF1'].dropna(), ax=axes,lags=50);

### plot pacf high ###

fig, axes = plt.subplots(figsize=(10, 5))
plot_pacf(train_high_set['High_sq_DIFF1'].dropna(), ax=axes,lags=50);


# In[68]:


### plot acf high ###

fig, axes = plt.subplots(figsize=(10, 5))
plot_acf(train_high_set['High_box_DIFF1'].dropna(), ax=axes,lags=50);

### plot pacf high ###

fig, axes = plt.subplots(figsize=(10, 5))
plot_pacf(train_high_set['High_box_DIFF1'].dropna(), ax=axes,lags=50);


# In[ ]:





# In[ ]:





# In[ ]:





# In[69]:


### plot acf low ###

fig, axes = plt.subplots(figsize=(10, 5))
plot_acf(train_low_set['Low_DIFF1'].dropna(), ax=axes,lags=50);


### plot pacf l0w ###

fig, axes = plt.subplots(figsize=(10, 5))
plot_pacf(train_low_set['Low_DIFF1'].dropna(), ax=axes,lags=50);


# In[70]:


### plot acf low ###

fig, axes = plt.subplots(figsize=(10, 5))
plot_acf(train_low_set['Low_log_DIFF1'].dropna(), ax=axes,lags=50);

### plot pacf low ###

fig, axes = plt.subplots(figsize=(10, 5))
plot_pacf(train_low_set['Low_log_DIFF1'].dropna(), ax=axes,lags=50);


# In[71]:


### plot acf low ###

fig, axes = plt.subplots(figsize=(10, 5))
plot_acf(train_low_set['Low_recip_DIFF1'].dropna(), ax=axes,lags=50);

### plot pacf low ###

fig, axes = plt.subplots(figsize=(10, 5))
plot_pacf(train_low_set['Low_recip_DIFF1'].dropna(), ax=axes,lags=50);


# In[72]:


### plot acf low ###

fig, axes = plt.subplots(figsize=(10, 5))
plot_acf(train_low_set['Low_sqrt_DIFF1'].dropna(), ax=axes,lags=50);

### plot pacf low ###

fig, axes = plt.subplots(figsize=(10, 5))
plot_pacf(train_low_set['Low_sqrt_DIFF1'].dropna(), ax=axes,lags=50);


# In[73]:


### plot acf low ###

fig, axes = plt.subplots(figsize=(10, 5))
plot_acf(train_low_set['Low_sqrt_DIFF1'].dropna(), ax=axes,lags=50);

### plot pacf low ###

fig, axes = plt.subplots(figsize=(10, 5))
plot_pacf(train_low_set['Low_sqrt_DIFF1'].dropna(), ax=axes,lags=50);


# In[74]:


### plot acf low ###

fig, axes = plt.subplots(figsize=(10, 5))
plot_acf(train_low_set['Low_sq_DIFF1'].dropna(), ax=axes,lags=50);

### plot pacf low ###

fig, axes = plt.subplots(figsize=(10, 5))
plot_pacf(train_low_set['Low_sq_DIFF1'].dropna(), ax=axes,lags=50);


# In[75]:


### plot acf low ###

fig, axes = plt.subplots(figsize=(10, 5))
plot_acf(train_low_set['Low_box_DIFF1'].dropna(), ax=axes,lags=50);

### plot pacf low ###

fig, axes = plt.subplots(figsize=(10, 5))
plot_pacf(train_low_set['Low_box_DIFF1'].dropna(), ax=axes,lags=50);


# ## ARIMA MODELS

# In[59]:


#!pip install pmdarima


# In[563]:


import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA
#from pmdarima import auto_arima


# In[ ]:





# In[544]:


### High Arima models pm ###
arima_high_log_110 = pm.ARIMA(order=(1, 1, 0)).fit( train_high_set['High_log'] )
arima_high_log_011 = pm.ARIMA(order=(0, 1, 1)).fit( train_high_set['High_log'] )
arima_high_log_111 = pm.ARIMA(order=(1, 1, 1)).fit( train_high_set['High_log'] )

arima_high_recip_110 = pm.ARIMA(order=(1, 1, 0)).fit( train_high_set['High_recip'] )
arima_high_recip_011 = pm.ARIMA(order=(0, 1, 1)).fit( train_high_set['High_recip'] )
arima_high_recip_111 = pm.ARIMA(order=(1, 1, 1)).fit( train_high_set['High_recip'] )

arima_high_sqrt_110 = pm.ARIMA(order=(1, 1, 0)).fit( train_high_set['High_sqrt'] )
arima_high_sqrt_011 = pm.ARIMA(order=(0, 1, 1)).fit( train_high_set['High_sqrt'] )
arima_high_sqrt_111 = pm.ARIMA(order=(1, 1, 1)).fit( train_high_set['High_sqrt'] )

arima_high_sq_110 = pm.ARIMA(order=(1, 1, 0)).fit( train_high_set['High_sq'] )
arima_high_sq_011 = pm.ARIMA(order=(0, 1, 1)).fit( train_high_set['High_sq'] )
arima_high_sq_111 = pm.ARIMA(order=(1, 1, 1)).fit( train_high_set['High_sq'] )

arima_high_box_110 = pm.ARIMA(order=(1, 1, 0)).fit( train_high_set['High_box'] )
arima_high_box_011 = pm.ARIMA(order=(0, 1, 1)).fit( train_high_set['High_box'] )
arima_high_box_111 = pm.ARIMA(order=(1, 1, 1)).fit( train_high_set['High_box'] )


# In[564]:


### High Arima models pm no fit ###
arima_high_log_110 = pm.ARIMA(order=(1, 1, 0))
arima_high_log_011 = pm.ARIMA(order=(0, 1, 1))
arima_high_log_111 = pm.ARIMA(order=(1, 1, 1))

arima_high_recip_110 = pm.ARIMA(order=(1, 1, 0))
arima_high_recip_011 = pm.ARIMA(order=(0, 1, 1))
arima_high_recip_111 = pm.ARIMA(order=(1, 1, 1))

arima_high_sqrt_110 = pm.ARIMA(order=(1, 1, 0))
arima_high_sqrt_011 = pm.ARIMA(order=(0, 1, 1))
arima_high_sqrt_111 = pm.ARIMA(order=(1, 1, 1))

arima_high_sq_110 = pm.ARIMA(order=(1, 1, 0))
arima_high_sq_011 = pm.ARIMA(order=(0, 1, 1))
arima_high_sq_111 = pm.ARIMA(order=(1, 1, 1))

arima_high_box_110 = pm.ARIMA(order=(1, 1, 0))
arima_high_box_011 = pm.ARIMA(order=(0, 1, 1))
arima_high_box_111 = pm.ARIMA(order=(1, 1, 1))


# In[545]:


### Low Arima models pm ###
arima_low_log_110 = pm.ARIMA(order=(1, 1, 0)).fit( train_low_set['Low_log'] )
arima_low_log_011 = pm.ARIMA(order=(0, 1, 1)).fit( train_low_set['Low_log'] )
arima_low_log_111 = pm.ARIMA(order=(1, 1, 1)).fit( train_low_set['Low_log'] )

arima_low_recip_110 = pm.ARIMA(order=(1, 1, 0)).fit( train_low_set['Low_recip'] )
arima_low_recip_011 = pm.ARIMA(order=(0, 1, 1)).fit( train_low_set['Low_recip'])
arima_low_recip_111 = pm.ARIMA(order=(1, 1, 1)).fit( train_low_set['Low_recip'] )

arima_low_sqrt_110 = pm.ARIMA(order=(1, 1, 0)).fit( train_low_set['Low_sqrt'] )
arima_low_sqrt_011 = pm.ARIMA(order=(0, 1, 1)).fit( train_low_set['Low_sqrt'] )
arima_low_sqrt_111 = pm.ARIMA(order=(1, 1, 1)).fit( train_low_set['Low_sqrt'] )

arima_low_sq_110 = pm.ARIMA(order=(1, 1, 0)).fit( train_low_set['Low_sq'] )
arima_low_sq_011 = pm.ARIMA(order=(0, 1, 1)).fit( train_low_set['Low_sq'] )
arima_low_sq_111 = pm.ARIMA(order=(1, 1, 1)).fit( train_low_set['Low_sq'] )

arima_low_box_110 = pm.ARIMA(order=(1, 1, 0)).fit( train_low_set['Low_box'] )
arima_low_box_011 = pm.ARIMA(order=(0, 1, 1)).fit( train_low_set['Low_box'] )
arima_low_box_111 = pm.ARIMA(order=(1, 1, 1)).fit( train_low_set['Low_box'] )


# In[565]:


### Low Arima models pm no fit ###
arima_low_log_110 = pm.ARIMA(order=(1, 1, 0))
arima_low_log_011 = pm.ARIMA(order=(0, 1, 1))
arima_low_log_111 = pm.ARIMA(order=(1, 1, 1))

arima_low_recip_110 = pm.ARIMA(order=(1, 1, 0))
arima_low_recip_011 = pm.ARIMA(order=(0, 1, 1))
arima_low_recip_111 = pm.ARIMA(order=(1, 1, 1))

arima_low_sqrt_110 = pm.ARIMA(order=(1, 1, 0))
arima_low_sqrt_011 = pm.ARIMA(order=(0, 1, 1))
arima_low_sqrt_111 = pm.ARIMA(order=(1, 1, 1))

arima_low_sq_110 = pm.ARIMA(order=(1, 1, 0))
arima_low_sq_011 = pm.ARIMA(order=(0, 1, 1))
arima_low_sq_111 = pm.ARIMA(order=(1, 1, 1))

arima_low_box_110 = pm.ARIMA(order=(1, 1, 0))
arima_low_box_011 = pm.ARIMA(order=(0, 1, 1))
arima_low_box_111 = pm.ARIMA(order=(1, 1, 1))


# In[ ]:





# In[566]:


### High models in a list ###
my_high_arima_list = [arima_high_log_110,arima_high_log_011 ,arima_high_log_111,
           arima_high_recip_110,arima_high_recip_011,arima_high_recip_111,
           arima_high_sqrt_110,arima_high_sqrt_011,arima_high_sqrt_111,
           arima_high_sq_110,arima_high_sq_011,arima_high_sq_111,
           arima_high_box_110,arima_high_box_011,arima_high_box_111]


# In[567]:


### Low models in a list ###
my_low_arima_list = [arima_low_log_110,arima_low_log_011 ,arima_low_log_111,
           arima_low_recip_110,arima_low_recip_011,arima_low_recip_111,
           arima_low_sqrt_110,arima_low_sqrt_011,arima_low_sqrt_111,
           arima_low_sq_110,arima_low_sq_011,arima_low_sq_111,
           arima_low_box_110,arima_low_box_011,arima_low_box_111]


# In[ ]:





# In[568]:


my_high_arima_names_list = ['high_log_110','high_log_011','high_log_111',
                         'high_recip_110','high_recip_011','high_recip_111',
                         'high_sqrt_110','high_sqrt_011','high_sqrt_111',
                         'high_sq_110','high_sq_011','high_sq_111',
                         'high_box_110','high_box_011','high_box_111']
                         
my_low_arima_names_list = ['low_log_110','low_log_011','low_log_111',
                         'low_recip_110','low_recip_011','low_recip_111',
                         'low_sqrt_110','low_sqrt_011','low_sqrt_111',
                         'low_sq_110','low_sq_011','low_sq_111',
                         'low_box_110','low_box_011','low_box_111'] 


# # my time series class

# In[571]:


from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_percentage_error,mean_absolute_error
class pre_model_selection:
    
    def __init__(self,model,model_name,model_order):
        
        self.model = model
        self.model_name = model_name
        self.model_order = model_order
        self.model_aic = model.aic()
        self.y = None
        self.yhat = None
        self.transformation_type = None
        self.yhat = None
        self.level = None
        self.AIC = None
        self.r_squared = None
        self.rmse = None
        self.mape = None
        self.mae = None
        
    ### get metrics ########################################################
    def get_r_squared(self):
        self.r_squared = round(r2_score(self.y,self.yhat),7)
        return self.r_squared,
    
    def get_rmse(self):
        self.rmse = round(mean_squared_error(self.y,self.yhat,squared=False),7)
        return self.rmse
        
    
    def get_mape(self):
        self.mape = round(mean_absolute_percentage_error(self.y,self.yhat),7)
        return  self.mape
    
    def get_mae(self):
        self.mae = round(mean_absolute_error(self.y,self.yhat),7)
        return self.mae
    
    def get_AIC(self):
      
        residuals = (np.array(self.y) - np.array(self.yhat))[0]
        rss = sum(residuals**2)
        k= self.model_order[0]+self.model_order[1]+self.model_order[2] + 1 # p + d + q + 1
        n = len(self.yhat)
        self.AIC = round( 2*k + n*( np.log( 2*(np.pi)*rss/n)+1),7 )
        
        return self.AIC
    
   

    
    
    
        
    ### get methods ########################################################
    def get_aic(self):
        return self.model.aic()
    
    def get_name(self):
        return self.model_name
    
    def get_model_order(self):
        return self.model_order
    
    def get_transformation_type(self):
        return self.transformation_type
    
    def get_level(self):
        return self.level
    
    def get_model(self):
        return self.model
    
    ### setter methods ######################################################
    def set_yhat(self,x):
        self.yhat = x
        
    def set_transformation_type(self,x):
        transformations = [None,'log','sqrt','sq','recipical','box']
        if x.lower() in transformations:
            self.transformation_type = x
        else:
            print(f'Invalid input, value must be {transformations}')
            
    def set_yhat(self,x):
        
        if self.transformation_type == 'log':
            self.yhat = invert_log_transformer(x)
        
        elif self.transformation_type == 'sqrt':
            self.yhat = invert_sqrt_transformer(x)
        
        elif self.transformation_type == 'sq':
            self.yhat = invert_sq_transformer(x)
        
        elif self.transformation_type == 'recipical':
            self.yhat = invert_recipical_transformer(x)
        
        elif self.transformation_type == 'box':
            if self.level == 'high':
                self.yhat = invert_boxcox_transformer_high(x,lambda_high)
            else:
                self.yhat = invert_boxcox_transformer_low(x,lambda_low)
        else:
            self.yhat = x
        
                
    def set_level(self,x):
        
        levels = ['high','low',None]
        
        if x.lower() in levels:
            self.level = x
        else:
            print(f'invalid input, levels must be {levels}')
            
    def set_y(self,x):
        self.y = x
            
    ######################################################
    
    def __str__(self):
        return f"####################\nMODEL:{self.model_name}\nOrder:{self.model_order}\nTransform:{self.transformation_type}\nLevel:{self.level}\n\n\nR_Square:{self.r_squared}\nAIC:{self.AIC}\nrmse:{self.rmse}\nmape:{self.mape}\nmae:{self.mae}\n##############################\n\n\n"
    
    


# In[ ]:





# In[ ]:





# ### best high models

# In[ ]:





# In[550]:


best_high_models_search = []
order_list = [(0,1,1),(1,1,0),(1,1,1)]
dummy = 0

for i,j in zip(my_high_arima_list,my_high_arima_names_list):
    index = dummy%3
    model = pre_model_selection(i,j,order_list[index])
    best_high_models_search.append(model)
    dummy = dummy + 1
        
from operator import attrgetter
best_high_models_search.sort(key = attrgetter('model_aic'), reverse = False)

for i in best_high_models_search:
    print(i.get_name())
    print(i.get_aic())


# In[290]:


# skip do not run
arima_high_log_011 = pm.ARIMA(order=(0, 1, 1)).fit( train_high_set['High_log'] )
arima_high_recip_011 = pm.ARIMA(order=(0, 1, 1)).fit( train_high_set['High_recip'] )
arima_high_sqrt_011 = pm.ARIMA(order=(0, 1, 1)).fit( train_high_set['High_sqrt'] )
arima_high_sq_011 = pm.ARIMA(order=(0, 1, 1)).fit( train_high_set['High_sq'] )
arima_high_box_011 = pm.ARIMA(order=(0, 1, 1)).fit( train_high_set['High_box'] )

best_high_models = [arima_high_log_011, 
                    arima_high_recip_011,
                    arima_high_sqrt_011,
                    arima_high_sq_011,
                    arima_high_box_011]

my_high_arima_names_list = ['high_log_011','high_recip_011','high_sqrt_011','high_sq_011','high_box_011']

best_high_models_final = []
order_list = [(0,1,1)]

for i,j in zip(best_high_models,my_high_arima_names_list):
    index = 0
    model = pre_model_selection(i,j,order_list[index])
    best_high_models_final.append(model)
        
from operator import attrgetter
best_high_models_final.sort(key = attrgetter('model_aic'), reverse = False)

for i in best_high_models_final:
    print(i.get_name())
    print(i.get_aic())


# In[ ]:





# In[ ]:



                         


# ### best low models

# In[551]:


best_low_models_search = []
order_list = [(0,1,1),(1,1,0),(1,1,1)]
dummy = 0

for i,j in zip(my_low_arima_list,my_low_arima_names_list):
    index = dummy%3
    model = pre_model_selection(i,j,order_list[index])
    best_low_models_search.append(model)
    dummy = dummy +1
        
from operator import attrgetter
best_low_models_search.sort(key = attrgetter('model_aic'), reverse = False)

for i in best_low_models_search:
    print(i.get_name())
    print(i.get_aic())


# In[292]:


# skip do not run 
arima_low_log_011 = pm.ARIMA(order=(0, 1, 1)).fit( train_low_set['Low_log'] )
arima_low_recip_011 = pm.ARIMA(order=(0, 1, 1)).fit( train_low_set['Low_recip'])
arima_low_sqrt_011 = pm.ARIMA(order=(0, 1, 1)).fit( train_low_set['Low_sqrt'] )
arima_low_sq_011 = pm.ARIMA(order=(0, 1, 1)).fit( train_low_set['Low_sq'] )
arima_low_box_011 = pm.ARIMA(order=(0, 1, 1)).fit( train_low_set['Low_box'] )

best_low_models = [arima_low_log_011,
                    arima_low_recip_011,
                    arima_low_sqrt_011,
                    arima_low_sq_011,
                    arima_low_box_011 ]

my_low_arima_names_list =  ['low_log_011','low_recip_011','low_sqrt_011','low_sq_011','low_box_011']

best_low_models_final = []
order_list = [(0,1,1)]

for i,j in zip(best_low_models,my_low_arima_names_list):
    index = 0
    model = pre_model_selection(i,j,order_list[index])
    best_low_models_final.append(model)
        
from operator import attrgetter
best_low_models_final.sort(key = attrgetter('model_aic'), reverse = False)

for i in best_low_models_final:
    print(i.get_name())
    print(i.get_aic())


# In[ ]:





# In[ ]:





# In[552]:


for i in best_high_models_search:
    i.set_level('high')
    

best_high_models_search[0].set_transformation_type('sqrt')
best_high_models_search[1].set_transformation_type('sqrt')
best_high_models_search[2].set_transformation_type('sqrt')
best_high_models_search[3].set_transformation_type('box')
best_high_models_search[4].set_transformation_type('box')
best_high_models_search[5].set_transformation_type('box')
best_high_models_search[6].set_transformation_type('recipical')
best_high_models_search[7].set_transformation_type('recipical')
best_high_models_search[8].set_transformation_type('recipical')
best_high_models_search[9].set_transformation_type('log')
best_high_models_search[10].set_transformation_type('log')
best_high_models_search[11].set_transformation_type('log')
best_high_models_search[12].set_transformation_type('sq')
best_high_models_search[13].set_transformation_type('sq')
best_high_models_search[14].set_transformation_type('sq')


# In[553]:


for i in best_low_models_search:
    i.set_level('low')
    

best_low_models_search[0].set_transformation_type('sqrt')
best_low_models_search[1].set_transformation_type('sqrt')
best_low_models_search[2].set_transformation_type('sqrt')
best_low_models_search[3].set_transformation_type('box')
best_low_models_search[4].set_transformation_type('box')
best_low_models_search[5].set_transformation_type('box')
best_low_models_search[6].set_transformation_type('recipical')
best_low_models_search[7].set_transformation_type('recipical')
best_low_models_search[8].set_transformation_type('recipical')
best_low_models_search[9].set_transformation_type('log')
best_low_models_search[10].set_transformation_type('log')
best_low_models_search[11].set_transformation_type('log')
best_low_models_search[12].set_transformation_type('sq')
best_low_models_search[13].set_transformation_type('sq')
best_low_models_search[14].set_transformation_type('sq')


# In[573]:


validation_high_names = ['High_sqrt','High_sqrt','High_sqrt',
                         'High_box','High_box','High_box',
                         'High_recip','High_recip','High_recip',
                         'High_log','High_log','High_log',
                         'High_sq','High_sq','High_sq']

validation_low_names = ['Low_sqrt','Low_sqrt','Low_sqrt',
                        'Low_box','Low_box','Low_box',
                        'Low_recip','Low_recip','Low_recip','Low_recip',
                        'Low_log','Low_log','Low_log',
                        'Low_sq','Low_sq','Low_sq']


# In[ ]:





# ## Cross Validation

# In[ ]:


### High Arima models pm ###
arima_high_log_110 = pm.ARIMA(order=(1, 1, 0)).fit( train_high_set['High_log'] )
arima_high_log_011 = pm.ARIMA(order=(0, 1, 1)).fit( train_high_set['High_log'] )
arima_high_log_111 = pm.ARIMA(order=(1, 1, 1)).fit( train_high_set['High_log'] )

arima_high_recip_110 = pm.ARIMA(order=(1, 1, 0)).fit( train_high_set['High_recip'] )
arima_high_recip_011 = pm.ARIMA(order=(0, 1, 1)).fit( train_high_set['High_recip'] )
arima_high_recip_111 = pm.ARIMA(order=(1, 1, 1)).fit( train_high_set['High_recip'] )

arima_high_sqrt_110 = pm.ARIMA(order=(1, 1, 0)).fit( train_high_set['High_sqrt'] )
arima_high_sqrt_011 = pm.ARIMA(order=(0, 1, 1)).fit( train_high_set['High_sqrt'] )
arima_high_sqrt_111 = pm.ARIMA(order=(1, 1, 1)).fit( train_high_set['High_sqrt'] )

arima_high_sq_110 = pm.ARIMA(order=(1, 1, 0)).fit( train_high_set['High_sq'] )
arima_high_sq_011 = pm.ARIMA(order=(0, 1, 1)).fit( train_high_set['High_sq'] )
arima_high_sq_111 = pm.ARIMA(order=(1, 1, 1)).fit( train_high_set['High_sq'] )

arima_high_box_110 = pm.ARIMA(order=(1, 1, 0)).fit( train_high_set['High_box'] )
arima_high_box_011 = pm.ARIMA(order=(0, 1, 1)).fit( train_high_set['High_box'] )
arima_high_box_111 = pm.ARIMA(order=(1, 1, 1)).fit( train_high_set['High_box'] )


# In[572]:


### High Arima models pm ###
arima_high_log_110 = pm.ARIMA(order=(1, 1, 0))
arima_high_log_011 = pm.ARIMA(order=(0, 1, 1))
arima_high_log_111 = pm.ARIMA(order=(1, 1, 1))

arima_high_recip_110 = pm.ARIMA(order=(1, 1, 0))
arima_high_recip_011 = pm.ARIMA(order=(0, 1, 1))
arima_high_recip_111 = pm.ARIMA(order=(1, 1, 1))

arima_high_sqrt_110 = pm.ARIMA(order=(1, 1, 0))
arima_high_sqrt_011 = pm.ARIMA(order=(0, 1, 1))
arima_high_sqrt_111 = pm.ARIMA(order=(1, 1, 1))

arima_high_sq_110 = pm.ARIMA(order=(1, 1, 0))
arima_high_sq_011 = pm.ARIMA(order=(0, 1, 1))
arima_high_sq_111 = pm.ARIMA(order=(1, 1, 1))

arima_high_box_110 = pm.ARIMA(order=(1, 1, 0))
arima_high_box_011 = pm.ARIMA(order=(0, 1, 1))
arima_high_box_111 = pm.ARIMA(order=(1, 1, 1))
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
### Low Arima models pm no fit ###
arima_low_log_110 = pm.ARIMA(order=(1, 1, 0))
arima_low_log_011 = pm.ARIMA(order=(0, 1, 1))
arima_low_log_111 = pm.ARIMA(order=(1, 1, 1))

arima_low_recip_110 = pm.ARIMA(order=(1, 1, 0))
arima_low_recip_011 = pm.ARIMA(order=(0, 1, 1))
arima_low_recip_111 = pm.ARIMA(order=(1, 1, 1))

arima_low_sqrt_110 = pm.ARIMA(order=(1, 1, 0))
arima_low_sqrt_011 = pm.ARIMA(order=(0, 1, 1))
arima_low_sqrt_111 = pm.ARIMA(order=(1, 1, 1))

arima_low_sq_110 = pm.ARIMA(order=(1, 1, 0))
arima_low_sq_011 = pm.ARIMA(order=(0, 1, 1))
arima_low_sq_111 = pm.ARIMA(order=(1, 1, 1))

arima_low_box_110 = pm.ARIMA(order=(1, 1, 0))
arima_low_box_011 = pm.ARIMA(order=(0, 1, 1))
arima_low_box_111 = pm.ARIMA(order=(1, 1, 1))


# In[557]:


best_high_models_search


# In[558]:


validation_high_names


# In[670]:


high_high_models = [arima_high_sqrt_110,arima_high_sqrt_011,arima_high_sqrt_111,
                    arima_high_box_110,arima_high_box_011,arima_high_box_111,
                    arima_high_recip_110,arima_high_recip_011,arima_high_recip_111,
                    arima_high_log_110,arima_high_log_011,arima_high_log_111,
                    arima_high_sq_110,arima_high_sq_011,arima_high_sq_111]


# In[671]:


low_train_models = [arima_low_sqrt_110,arima_low_sqrt_011,arima_low_sqrt_111,
                    arima_low_box_110,arima_low_box_011,arima_low_box_111,
                    arima_low_recip_110,arima_low_recip_011,arima_low_recip_111,
                    arima_low_log_110,arima_low_log_011,arima_low_log_111,
                    arima_low_sq_110,arima_low_sq_011,arima_low_sq_111]


# In[672]:


validation_high_names


# In[673]:


def rolling_validation_one_step(model,train,test):
    
    Train = train.copy()
    Test = test.copy()
    predictions = []
    size = len(test)
    
    model.fit(Train)
    
    for i in range(size):
        pred= model.predict(1)
        predictions.append(pred)
        Train = Train.append(Test[i:i+1])
        model.fit(Train)
        
    return predictions


# In[674]:


# new cross validaion High
high_predictions_collection = []

for i,j in zip(high_train_models,validation_high_names):
    high_predictions_collection.append(rolling_validation_one_step(i,train_high_set[j],validation_high_set[j]))


# In[675]:


# new cross validaion  Low
low_predictions_collection = []

for i,j in zip(low_train_models,validation_low_names):
    low_predictions_collection.append(rolling_validation_one_step(i,train_low_set[j],validation_low_set[j]))


# In[677]:


validation_high_names


# In[ ]:


invert_log_transformer
invert_recipical_transformer
invert_sqrt_transformer
invert_sq_transformer
invert_boxcox_transformer_high  
invert_boxcox_transformer_low 


# In[664]:


invert_sqrt_transformer("(high_predictions_collection[0])")

mean_squared_error(y,yhat,squared=False)


# In[680]:


### high
#################################################################################################
rmse_high_sqrt110 = mean_squared_error(validation_high_set['High'],
                          invert_sqrt_transformer(np.array(high_predictions_collection[0])),
                          squared=False)
rmse_high_sqrt011 = mean_squared_error(validation_high_set['High'],
                          invert_sqrt_transformer(np.array(high_predictions_collection[1])),
                          squared=False)
rmse_high_sqrt111 = mean_squared_error(validation_high_set['High'],
                          invert_sqrt_transformer(np.array(high_predictions_collection[2])),
                          squared=False)
#################################################################################################
rmse_high_box110 = mean_squared_error(validation_high_set['High'],
                          invert_boxcox_transformer_high(np.array(high_predictions_collection[3]),lambda_high),
                          squared=False)
rmse_high_box011 = mean_squared_error(validation_high_set['High'],
                          invert_boxcox_transformer_high(np.array(high_predictions_collection[4]),lambda_high),
                          squared=False)
rmse_high_box111 = mean_squared_error(validation_high_set['High'],
                          invert_boxcox_transformer_high(np.array(high_predictions_collection[5]),lambda_high),
                          squared=False)
#################################################################################################
rmse_high_recip110 = mean_squared_error(validation_high_set['High'],
                          invert_recipical_transformer(np.array(high_predictions_collection[6])),
                          squared=False)
rmse_high_recip011 = mean_squared_error(validation_high_set['High'],
                          invert_recipical_transformer(np.array(high_predictions_collection[7])),
                          squared=False)
rmse_high_recip111 = mean_squared_error(validation_high_set['High'],
                          invert_recipical_transformer(np.array(high_predictions_collection[8])),
                          squared=False)
#################################################################################################
rmse_high_log110 = mean_squared_error(validation_high_set['High'],
                          invert_log_transformer(np.array(high_predictions_collection[9])),
                          squared=False)
rmse_high_log011 = mean_squared_error(validation_high_set['High'],
                          invert_log_transformer(np.array(high_predictions_collection[10])),
                          squared=False)
rmse_high_log111 = mean_squared_error(validation_high_set['High'],
                          invert_log_transformer(np.array(high_predictions_collection[11])),
                          squared=False)
#################################################################################################
rmse_high_sq110 = mean_squared_error(validation_high_set['High'],
                          invert_sq_transformer(np.array(high_predictions_collection[12])),
                          squared=False)
rmse_high_sq011 = mean_squared_error(validation_high_set['High'],
                          invert_sq_transformer(np.array(high_predictions_collection[13])),
                          squared=False)
rmse_high_sq111 = mean_squared_error(validation_high_set['High'],
                          invert_sq_transformer(np.array(high_predictions_collection[14])),
                          squared=False)
#################################################################################################
high_rmse = [rmse_high_sqrt110,rmse_high_sqrt011,rmse_high_sqrt111,
             rmse_high_box110,rmse_high_box011,rmse_high_box111,
             rmse_high_recip110,rmse_high_recip011,rmse_high_recip111,
             rmse_high_log110,rmse_high_log011,rmse_high_log111,
             rmse_high_sq110,rmse_high_sq011,rmse_high_sq111]

high_rmse_name = ['rmse_high_sqrt110','rmse_high_sqrt011','rmse_high_sqrt111',
                  'rmse_high_box110','rmse_high_box011','rmse_high_box111',
                  'rmse_high_recip110','rmse_high_recip011','rmse_high_recip111',
                  'rmse_high_log110','rmse_high_log011','rmse_high_log111',
                  'rmse_high_sq110','rmse_high_sq011','rmse_high_sq111']
#################################################################################################
High_rmse_score_dictionary = {}

for i,j in zip(high_rmse_name,high_rmse):
    High_rmse_score_dictionary[i] = j
#################################################################################################


# In[681]:


### Low
#################################################################################################
rmse_low_sqrt110 = mean_squared_error(validation_low_set['Low'],
                          invert_sqrt_transformer(np.array(low_predictions_collection[0])),
                          squared=False)
rmse_low_sqrt011 = mean_squared_error(validation_low_set['Low'],
                          invert_sqrt_transformer(np.array(low_predictions_collection[1])),
                          squared=False)
rmse_low_sqrt111 = mean_squared_error(validation_low_set['Low'],
                          invert_sqrt_transformer(np.array(high_predictions_collection[2])),
                          squared=False)
#################################################################################################
rmse_low_box110 = mean_squared_error(validation_low_set['Low'],
                          invert_boxcox_transformer_low(np.array(low_predictions_collection[3]),lambda_low),
                          squared=False)
rmse_low_box011 = mean_squared_error(validation_low_set['Low'],
                          invert_boxcox_transformer_low(np.array(low_predictions_collection[4]),lambda_low),
                          squared=False)
rmse_low_box111 = mean_squared_error(validation_low_set['Low'],
                          invert_boxcox_transformer_low(np.array(low_predictions_collection[5]),lambda_low),
                          squared=False)
#################################################################################################
rmse_low_recip110 = mean_squared_error(validation_low_set['Low'],
                          invert_recipical_transformer(np.array(low_predictions_collection[6])),
                          squared=False)
rmse_low_recip011 = mean_squared_error(validation_low_set['Low'],
                          invert_recipical_transformer(np.array(low_predictions_collection[7])),
                          squared=False)
rmse_low_recip111 = mean_squared_error(validation_low_set['Low'],
                          invert_recipical_transformer(np.array(low_predictions_collection[8])),
                          squared=False)
#################################################################################################
rmse_low_log110 = mean_squared_error(validation_low_set['Low'],
                          invert_log_transformer(np.array(low_predictions_collection[9])),
                          squared=False)
rmse_low_log011 = mean_squared_error(validation_low_set['Low'],
                          invert_log_transformer(np.array(low_predictions_collection[10])),
                          squared=False)
rmse_low_log111 = mean_squared_error(validation_low_set['Low'],
                          invert_log_transformer(np.array(low_predictions_collection[11])),
                          squared=False)
#################################################################################################
rmse_low_sq110 = mean_squared_error(validation_low_set['Low'],
                          invert_sq_transformer(np.array(low_predictions_collection[12])),
                          squared=False)
rmse_low_sq011 = mean_squared_error(validation_low_set['Low'],
                          invert_sq_transformer(np.array(low_predictions_collection[13])),
                          squared=False)
rmse_low_sq111 = mean_squared_error(validation_low_set['Low'],
                          invert_sq_transformer(np.array(low_predictions_collection[14])),
                          squared=False)
#################################################################################################
low_rmse = [rmse_low_sqrt110,rmse_low_sqrt011,rmse_low_sqrt111,
             rmse_low_box110,rmse_low_box011,rmse_low_box111,
             rmse_low_recip110,rmse_low_recip011,rmse_low_recip111,
             rmse_low_log110,rmse_low_log011,rmse_low_log111,
             rmse_low_sq110,rmse_low_sq011,rmse_low_sq111]

low_rmse_name = ['rmse_low_sqrt110','rmse_low_sqrt011','rmse_low_sqrt111',
                  'rmse_low_box110','rmse_low_box011','rmse_low_box111',
                  'rmse_low_recip110','rmse_low_recip011','rmse_low_recip111',
                  'rmse_low_log110','rmse_low_log011','rmse_low_log111',
                  'rmse_low_sq110','rmse_low_sq011','rmse_low_sq111']
#################################################################################################
Low_rmse_score_dictionary = {}

for i,j in zip(low_rmse_name,low_rmse):
    Low_rmse_score_dictionary[i] = j
#################################################################################################


# In[682]:


High_rmse_score_dictionary


# In[683]:


Low_rmse_score_dictionary


# In[ ]:


# rmse_high_recip011': 0.004013311126382556

# rmse_low_sq011': 0.003965062889338473


# In[ ]:





# In[ ]:





# In[ ]:


for i,j in zip(best_high_models_search,validation_high_names):
    
    cv = model_selection.RollingForecastCV(h=1,step=1,initial=3) #starts on 2019-08-30(3 days ahead because it gave nan)
    predictions = cross_val_predict( i.get_model() , validation_high_set[j], cv=cv, return_raw_predictions=True)
    
    yhat = predictions[3:].copy() # starts on 2019-08-30
    y = validation_high_set['High'][3:].copy() # starts on 2019-08-30
    
    i.set_yhat(yhat)
    i.set_y(y)
    i.get_r_squared()
    i.get_rmse()
    i.get_mape()
    i.get_mae()
    i.get_AIC()


# In[61]:


for i,j in zip(best_high_models_search,validation_high_names):
    
    cv = model_selection.RollingForecastCV(h=1,step=1,initial=3) #starts on 2019-08-30(3 days ahead because it gave nan)
    predictions = cross_val_predict( i.get_model() , validation_high_set[j], cv=cv, return_raw_predictions=True)
    
    yhat = predictions[3:].copy() # starts on 2019-08-30
    y = validation_high_set['High'][3:].copy() # starts on 2019-08-30
    
    i.set_yhat(yhat)
    i.set_y(y)
    i.get_r_squared()
    i.get_rmse()
    i.get_mape()
    i.get_mae()
    i.get_AIC()


# In[75]:


best_high_models_search.sort(key = attrgetter('AIC'), reverse = False)
print('Top 5 best AIC\n')
for i in best_high_models_search[0:5]:
    print(i.get_name())
    print(i.get_AIC())
    print()


# In[76]:


best_high_models_search.sort(key = attrgetter('r_squared'), reverse = True)
print('Top 5 best r_squared\n')
for i in best_high_models_search[0:5]:
    print(i.get_name())
    print(i.get_r_squared())
    print()


# In[62]:


best_high_models_search.sort(key = attrgetter('rmse'), reverse = False)
print('Top 5 best rmse\n')
for i in best_high_models_search[0:5]:
    print(i.get_name())
    print(i.get_rmse())
    print()


# In[78]:


best_high_models_search.sort(key = attrgetter('mape'), reverse = False)
print('Top 5 best mape\n')
for i in best_high_models_search[0:5]:
    print(i.get_name())
    print(i.get_mape())
    print()


# In[79]:


best_high_models_search.sort(key = attrgetter('mae'), reverse = False)
print('Top 5 best mae\n')
for i in best_high_models_search[0:5]:
    print(i.get_name())
    print(i.get_mae())
    print()


# In[ ]:





# In[63]:


for i,j in zip(best_low_models_search,validation_low_names):
    
    cv = model_selection.RollingForecastCV(h=1,step=1,initial=3) #starts on 2019-08-30(3 days ahead because it gave nan)
    predictions = cross_val_predict( i.get_model() , validation_low_set[j], cv=cv, return_raw_predictions=True)
    
    yhat = predictions[3:].copy() # starts on 2019-08-30
    y = validation_low_set['Low'][3:].copy() # starts on 2019-08-30
    
    i.set_yhat(yhat)
    i.set_y(y)
    i.get_r_squared()
    i.get_rmse()
    i.get_mape()
    i.get_mae()
    i.get_AIC()


# In[87]:


best_low_models_search.sort(key = attrgetter('AIC'), reverse = False)
print('Top 5 best AIC\n')
for i in best_low_models_search[0:5]:
    print(i.get_name())
    print(i.get_AIC())


# In[82]:


best_low_models_search.sort(key = attrgetter('r_squared'), reverse = True)
print('Top 5 r_squared\n')
for i in best_low_models_search[0:5]:
    print(i.get_name())
    print(i.get_r_squared())


# In[64]:


best_low_models_search.sort(key = attrgetter('rmse'), reverse = False)
print('Top 5 best rmse\n')
for i in best_low_models_search[0:5]:
    print(i.get_name())
    print(i.get_rmse())


# In[84]:


best_low_models_search.sort(key = attrgetter('mape'), reverse = False)
print('Top 5 best mape\n')
for i in best_low_models_search[0:5]:
    print(i.get_name())
    print(i.get_mape())


# In[86]:


best_low_models_search.sort(key = attrgetter('mae'), reverse = False)
print('Top 5 best mae\n')
for i in best_low_models_search[0:5]:
    print(i.get_name())
    print(i.get_mae())
    print()


# # Dianostic Testing

# In[66]:


#best_high_models_search # (0,1,1) recip


# In[67]:


#best_low_models_search # (0,1,1) sqrt


# In[ ]:


# rmse_high_recip011': 0.004013311126382556

# rmse_low_sq011': 0.003965062889338473


# In[684]:


best_low_model2 = pm.ARIMA(order=(0, 1, 1)).fit( low_set['Low_sq'] )


# In[685]:


best_low_model2.plot_diagnostics(figsize=(20,6));


# In[686]:


#perform Ljung-Box test on residuals with lag=5
sm.stats.acorr_ljungbox(best_low_model2.resid(), lags=[1], return_df=True)


# In[687]:


best_low_model2.summary()


# In[ ]:





# In[90]:


import statsmodels.api as sm
import matplotlib.pyplot as plt


# In[440]:


best_high_model = pm.ARIMA(order=(0, 1, 1)).fit( high_set['High_recip'] )


# In[214]:


best_high_model.plot_diagnostics(figsize=(20,6));


# In[302]:


#perform Ljung-Box test on residuals with lag=5
sm.stats.acorr_ljungbox(best_high_model.resid(), lags=[1], return_df=True)


# In[220]:


best_high_model.summary()


# In[ ]:





# In[441]:


best_low_model = pm.ARIMA(order=(0, 1, 1)).fit( low_set['Low_sqrt'] )


# In[222]:


best_low_model.plot_diagnostics(figsize=(20,6));


# In[301]:


#perform Ljung-Box test on residuals with lag=5
sm.stats.acorr_ljungbox(best_low_model.resid(), lags=[1], return_df=True)


# In[ ]:





# In[227]:


best_low_model.summary()


# # Forecast

# In[750]:


# rmse_high_recip011': 0.004013311126382556

# rmse_low_sq011': 0.003965062889338473


# In[711]:


def rolling_validation_one_step(model,train,test):
    
    Train = train.copy()
    Test = test.copy()
    predictions = []
    confindent_interval = []
    size = len(test)
    
    model.fit(Train)
    
    for i in range(size):
        pred,CI= model.predict(1,return_conf_int=True)
        predictions.append(pred)
        confindent_interval.append(CI)
        Train = Train.append(Test[i:i+1])
        model.fit(Train)
        
    return invert_recipical_transformer(np.array(predictions)),invert_recipical_transformer(np.array(confindent_interval))


# In[722]:


best_model_high1 = pm.ARIMA(order=(0, 1, 1))

pred_high,CI_high = rolling_validation_one_step(model=best_model_high1,
                                        train=high_set['High_recip'],
                                        test=test_high_set['High_recip'])


# In[723]:


date_test = pd.date_range('2021-07-27','2021-11-29',freq='B')


# In[748]:


results_high1 = pd.DataFrame({'Date':date_test,
                              'Actual':test_high_set['High'],
                              'Predictions':pred_high.flatten(),
                              'Confindent Interval': list(CI_high)})

results_high1.set_index('Date', inplace=True)


# In[749]:


results_high1


# In[751]:


def rolling_validation_one_step(model,train,test):
    
    Train = train.copy()
    Test = test.copy()
    predictions = []
    confindent_interval = []
    size = len(test)
    
    model.fit(Train)
    
    for i in range(size):
        pred,CI= model.predict(1,return_conf_int=True)
        predictions.append(pred)
        confindent_interval.append(CI)
        Train = Train.append(Test[i:i+1])
        model.fit(Train)
        
    return invert_sq_transformer(np.array(predictions)),invert_sq_transformer(np.array(confindent_interval))


# In[752]:


best_model_low1 = pm.ARIMA(order=(0, 1, 1))

pred_low,CI_low = rolling_validation_one_step(model=best_model_low1,
                                        train=low_set['Low_sq'],
                                        test=test_low_set['Low_sq'])


# In[753]:


date_test = pd.date_range('2021-07-27','2021-11-29',freq='B')


# In[754]:


results_low1 = pd.DataFrame({'Date':date_test,
                              'Actual':test_low_set['Low'],
                              'Predictions':pred_low.flatten(),
                              'Confindent Interval': list(CI_low)})

results_low1.set_index('Date', inplace=True)


# In[755]:


results_low1


# In[756]:


fig, axes = plt.subplots(figsize = (16,6),dpi=200)

results_high1[['Actual','Predictions']].plot(ax=axes);


# In[757]:


fig, axes = plt.subplots(figsize = (16,6),dpi=200)

results_low1[['Actual','Predictions']].plot(ax=axes);


# In[758]:


mean_squared_error(results_high1['Actual'],results_high1['Predictions'],squared=False)


# In[759]:


mean_squared_error(results_low1['Actual'],results_low1['Predictions'],squared=False)


# In[966]:


results_high1[89:90]


# In[967]:


results_low1[89:90]


# In[538]:


ths = test_high_set['High_recip'].copy()
pred = []
confindent_inteval = []
new_train =Train
for i in range(len(ths)):
    pred1,CI = model.predict(1,return_conf_int=True)
    pred.append(pred1)
    confindent_inteval.append(CI)
    new_train = new_train.append(ths[i:i+1])
    model = pm.ARIMA(order=(0, 1, 1)).fit(new_train)
    


# In[ ]:





# In[523]:


high_set['High_recip']


# In[540]:


len(pred)


# In[539]:


for i in pred:
    print(invert_recipical_transformer(i))


# In[541]:


results_high.head()


# In[470]:


hs = high_set['High_recip'].copy()
hs


# In[471]:


ths = test_high_set['High_recip'].copy()
ths


# In[490]:


hs.append(ths[0:1])


# In[489]:


ths[0:2]


# In[463]:


ths[2]


# In[493]:


ths[0:0+1]


# #  $\color{red}{\textbf{6 Outliers}}$

# In[ ]:





# In[ ]:





# In[289]:


date_residuals = pd.date_range('2012-01-02','2021-07-26',freq='B')


# In[290]:


# residuals High dataframe
residuals_high_df = pd.DataFrame({'Date':date_residuals,'residuals':best_high_model.resid()})
residuals_high_df['Date']  = pd.to_datetime( residuals_high_df['Date'] )
residuals_high_df.set_index('Date', inplace=True)


# In[291]:


residuals_high_df.head()


# In[292]:


UB_high = residuals_high_df['residuals'].mean() + 2*residuals_high_df['residuals'].std()
LB_high = residuals_high_df['residuals'].mean() - 2*residuals_high_df['residuals'].std()

residuals_high_df['residuals'].describe()


# In[293]:


residuals_high_df[residuals_high_df['residuals'] < LB_high]


# In[294]:


residuals_high_df[residuals_high_df['residuals'] > UB_high]


# In[ ]:





# In[ ]:





# In[ ]:





# In[295]:


# residuals Low dataframe
residuals_low_df = pd.DataFrame({'Date':date_residuals,'residuals':best_low_model.resid()})
residuals_low_df['Date']  = pd.to_datetime( residuals_low_df['Date'] )
residuals_low_df.set_index('Date', inplace=True)


# In[296]:


residuals_low_df.head()


# In[297]:


UB_low = residuals_low_df['residuals'].mean() + 2*residuals_low_df['residuals'].std()
LB_low = residuals_low_df['residuals'].mean() - 2*residuals_low_df['residuals'].std()

residuals_low_df['residuals'].describe()


# In[298]:


residuals_low_df[residuals_low_df['residuals'] < LB_low]


# In[299]:


residuals_low_df[residuals_low_df['residuals'] > UB_low]


# In[300]:


from scipy.stats import shapiro

#perform Shapiro-Wilk test
shapiro(residuals_high_df['residuals'])


# # $\color{red}{\textbf{6 Conclusion}}$

# In[439]:


date_test = pd.date_range('2021-07-30','2021-11-29',freq='B')


# In[321]:


best_high_model # recipical


# In[322]:


best_low_model


# In[ ]:





# In[366]:


cv = model_selection.RollingForecastCV(h=1,step=1,initial=3) #

predictions = cross_val_predict( best_high_model , test_high_set['High_recip'], cv=cv, return_raw_predictions=True)


# In[367]:


predictions_high = predictions[3:] # test_high_set[3:]


# In[368]:


results_high = pd.DataFrame({'Date':date_test,
                            'Actual':list(test_high_set[3:]['High']),
                            'Predictions':(list(invert_recipical_transformer(predictions_high).flatten()))})

results_high.set_index('Date', inplace=True)


# In[369]:


results_high


# In[370]:


fig, axes = plt.subplots(figsize = (16,6),dpi=200)

results_high.plot(ax=axes);


# In[372]:


mean_squared_error(results_high['Actual'],results_high['Predictions'],squared=False)


# In[373]:


cv = model_selection.RollingForecastCV(h=1,step=1,initial=3) #

predictions2 = cross_val_predict( best_low_model , test_low_set['Low_sqrt'], cv=cv, return_raw_predictions=True)


# In[374]:


predictions_low = predictions2[3:] 


# In[375]:


results_low = pd.DataFrame({'Date':date_test,
                            'Actual':list(test_low_set[3:]['Low']),
                            'Predictions':(list(invert_sqrt_transformer(predictions_low).flatten()))})

results_low.set_index('Date', inplace=True)


# In[376]:


fig, axes = plt.subplots(figsize = (16,6),dpi=200)

results_low.plot(ax=axes);


# In[377]:


mean_squared_error(results_low['Actual'],results_low['Predictions'],squared=False)


# In[381]:


results_high.head(),results_low.head()


# # $\color{red}{\textbf{7 Weaknesses}}$

# In[ ]:


residuals = (np.array(self.y) - np.array(self.yhat))[0]
        rss = sum(residuals**2)
        k= self.model_order[0]+self.model_order[1]+self.model_order[2] + 1 # p + d + q + 1
        n = len(self.yhat)
        self.AIC = round( 2*k + n*( np.log( 2*(np.pi)*rss/n)+1),7 )


# In[348]:


invert_recipical_transformer(predictions_high)


# In[407]:


x,y = best_high_model.predict(10,return_conf_int=True,alpha=0.50)
invert_recipical_transformer(x),invert_recipical_transformer(y)


# In[409]:


x,y = best_low_model.predict(10,return_conf_int=True,alpha=0.50)
invert_sqrt_transformer(x),invert_sqrt_transformer(y)


# In[386]:


test_low_set.head()


# In[391]:


results_high.head()


# In[608]:


# custom rolling function

def rolling_cv(train_set,min_train_size,horizon):
    
    for i in range(len(train_set)-min_train_size-horizon+1):
        split_train = train_set[:min_train_size+i]
        split_val = train_set[min_train_size+i:min_train_size+horizon]
        yield split_train, split_val


# In[616]:


# valid score metric
def cross_val_score(model,train,cv,metric):
    
    cv_scores = []
    predictions = []
    for cv_train, cv_test in cv:
        model.fit(cv_train)
        preds = model.predict(1)
        predictions.append(preds)
        score = metric(y_true=invert_recipical_transformer(cv_test),y_pred=invert_recipical_transformer(preds))
        cv_scores.append(score)
        
    return cv_scores,predictions


# In[617]:


Train = high_set['High_recip'].copy()
Test =  test_high_set['High_recip'].copy()
model = pm.ARIMA(order=(0, 1, 1))


# In[618]:


size_Train = len(Train)
size_Train


# In[619]:


size_Test = len(Test)
size_Test


# In[620]:


my_set = Train.append(Test)
my_set


# In[647]:


def rolling_validation_one_step(model,train,test):
    
    Train = train.copy()
    Test = test.copy()
    predictions = []
    size = len(test)
    
    model.fit(Train)
    
    for i in range(size):
        pred= model.predict(1)
        predictions.append(pred)
        Train = Train.append(Test[i:i+1])
        model.fit(Train)
        
    return predictions   


# In[645]:


model = pm.ARIMA(order=(0, 1, 1))
Train = high_set['High_recip'].copy()
Test = test_high_set['High_recip'].copy()

x = rolling_validation_one_step(model,Train,Test)


# In[646]:


invert_recipical_transformer(np.array(x))


# In[642]:


invert_recipical_transformer(np.array(x))


# In[ ]:


ths = test_high_set['High_recip'].copy()
pred = []
confindent_inteval = []
new_train =Train
for i in range(len(ths)):
    pred1,CI = model.predict(1,return_conf_int=True)
    pred.append(pred1)
    confindent_inteval.append(CI)
    new_train = new_train.append(ths[i:i+1])
    model = pm.ARIMA(order=(0, 1, 1)).fit(new_train)


# In[ ]:





# In[ ]:


### High Arima models pm ###
arima_high_log_110 = pm.ARIMA(order=(1, 1, 0)).fit( train_high_set['High_log'] )
arima_high_log_011 = pm.ARIMA(order=(0, 1, 1)).fit( train_high_set['High_log'] )
arima_high_log_111 = pm.ARIMA(order=(1, 1, 1)).fit( train_high_set['High_log'] )

arima_high_recip_110 = pm.ARIMA(order=(1, 1, 0)).fit( train_high_set['High_recip'] )
arima_high_recip_011 = pm.ARIMA(order=(0, 1, 1)).fit( train_high_set['High_recip'] )
arima_high_recip_111 = pm.ARIMA(order=(1, 1, 1)).fit( train_high_set['High_recip'] )

arima_high_sqrt_110 = pm.ARIMA(order=(1, 1, 0)).fit( train_high_set['High_sqrt'] )
arima_high_sqrt_011 = pm.ARIMA(order=(0, 1, 1)).fit( train_high_set['High_sqrt'] )
arima_high_sqrt_111 = pm.ARIMA(order=(1, 1, 1)).fit( train_high_set['High_sqrt'] )

arima_high_sq_110 = pm.ARIMA(order=(1, 1, 0)).fit( train_high_set['High_sq'] )
arima_high_sq_011 = pm.ARIMA(order=(0, 1, 1)).fit( train_high_set['High_sq'] )
arima_high_sq_111 = pm.ARIMA(order=(1, 1, 1)).fit( train_high_set['High_sq'] )

arima_high_box_110 = pm.ARIMA(order=(1, 1, 0)).fit( train_high_set['High_box'] )
arima_high_box_011 = pm.ARIMA(order=(0, 1, 1)).fit( train_high_set['High_box'] )
arima_high_box_111 = pm.ARIMA(order=(1, 1, 1)).fit( train_high_set['High_box'] )

