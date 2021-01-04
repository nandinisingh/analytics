#!/usr/bin/env python
# coding: utf-8

# ### Importing stuff

# In[1]:


import numpy as np
import pandas as pd
import datetime

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Getting data

# #### Getting data using Datareader

# In[5]:


import pandas_datareader
from pandas_datareader import data


# In[6]:


netflix = data.DataReader("NFLX",
                         start = '2009-1-1',
                         end = '2019-11-08',
                         data_source='yahoo')['Adj Close']

df_nf = pd.DataFrame(netflix)

df_nf.reset_index(inplace = True)
df_nf['Date'] = pd.to_datetime(df_nf.Date)

df_nf.set_index('Date', inplace = True)
df_nf.head()


# In[7]:


print(plt.style.available)


# In[8]:


plt.style.use('fivethirtyeight')
df_nf.plot(title='Netflix Adj. Closing Price', figsize = (10,5), color = 'green')

plt.show()


# In[9]:


# Convert the adjusted closing prices to cumulative returns

# Take percentage change of stock price of Netflix
df_ret = df_nf.pct_change()
df_ret
#Calculate the cumulative product 
# In cumulative product, the length of returned series is same as input series and every element is equal
#to the product of current and all previous values

plt.style.use('bmh')
((1 + df_ret).cumprod() - 1).plot(title = 'Netflix Cumulative Returns', figsize = (10,5), color = 'purple')


# In[10]:


# Plot the returns which is the percentage change in adjusted closing price

plt.style.use('tableau-colorblind10')
df_nf.pct_change().plot(figsize=(10,5), color = 'green')


# #### Getting data using an API key

# In[11]:


# source: https://github.com/RomelTorres/alpha_vantage
# source: https://alpha-vantage.readthedocs.io/en/latest/
        
# You will need to obtain a new API key from Aplha Vantage

# In bash shell type command below:
#     pip install alpha_vantage


# In[12]:


# Pull intraday data, example using Disney stock

# from alpha_vantage.timeseries import TimeSeries


# #### Uploading data

# In[13]:


# Google Trends website, search for terms Golf, Soccer, Tennis, Hockey and Baseball

sports = pd.read_csv('sports.csv', skiprows = 2)
df_sp = pd.DataFrame(sports)

col_names = ['Month','Golf','Soccer','Tennis','Hockey','Baseball']

df_sp.columns = col_names

df_sp.head()


# In[14]:


df_sp.tail()


# In[15]:


df_sp.isna().sum()


# In[16]:


df_sp.set_index('Month', inplace = True)
df_sp.describe()


# In[17]:


plt.style.use('classic')
df_sp.plot(figsize = (12,6), fontsize = 10, linewidth = 3, subplots = True)
plt.tight_layout()


# ### Time Series Data and Relationships

# We will look at the following 5 datasets:
# 
# 1. Google Trends. It's basically a term search count of the word "vacation".
# 2. Retail Furniture and furnishing sales data in millions of dollars. 
# 3. Adjusted clothes stock price for Bank of America. 
# 4. Adjusted clothes stock price for JP Morgan bank.
# 5. The monthly average temperature in Fahrenheit for St. Louis.

# #### Example 1: Vacation dataset

# In[18]:


vacation = pd.read_csv('vacation.csv',skiprows = 2)

df_v = pd.DataFrame(vacation)
# df_v.rename(columns = {'vacation: (United States)':'vacation'}, inplace = True)
col_names = ['month','num_search_vacation']
df_v.columns = col_names
df_v.head()


# In[19]:


# Check for missing values
df_v.isna().sum()


# In[20]:


from datetime import datetime

df_v['month'] = pd.to_datetime(df_v['month'])
df_v.set_index('month', inplace = True)


# In[21]:


df_v.head(3)


# In[22]:


# Provide the descriptive (summary) statistics
# Generate descriptive statistics that summarize the central tendency, dispersion and shape of the dataset's
# distribution, excluding NaN values.
# Percentile values (quantile, 1, 2, 3) on numeric values

df_v.describe()


# In[23]:


# Calculate the median value (middle value), which is the 50th percentile value, quantile 2
# Mean > median implies that the data is right skewed
# Mean < median implies that the data is left skewed
df_v.median()


# In[24]:


# Plot the time series of google searches of the word "vacation"

plt.style.use('seaborn-deep')

# #create Figure (empty Canvas)
# fig = plt.figure()

# # Add set of axes to the figure
# ax = fig.add_axes([0.1,0.1,0.8,0.9]) # left, bottom, width, height (range 0 to 1)

ax = df_v.plot(color = 'coral', grid = True, linewidth = 3, figsize=(10,5))

ax.set_xlabel('Year') # Notice the use of set_ to begin methods
ax.set_ylabel('Number of Searches')
ax.set_title('Google Trend of the word "vacation"')

plt.show()


# Visually inspecting the time series above you can see that it trends downwards and then stabilizes around 2013. There is also periodic patterns or cycles with the low points in the search for the word "vacation", mostly occurring in October of each year, though occasionally it is in November as well. There's a notable spike in June 2015 with 75 counts of the search term "vacation". The grid lines help us to see that the pattern repeats every year.

# In[25]:


# Check the options for fonts, lines, styles

print(plt.style.available)


# In[26]:


# Plot histogram (frequenct of counts), change num of bins to see different plots
df_v.plot(kind = 'hist', bins = 50, color = 'pink', grid = True)


# In[27]:


# Calculate Kernel Density Plot
# A density plot shows the distribution of the data over a continuous interval.
# Kernel Density Plot is a better way to display the distribution because it's not affected by the number of bins 
# used (each bar used in a typical histogram).

df_v.plot(kind = 'density', color = 'blue', grid = True, linewidth = 3, fontsize = 10)


# #### Example 2: Furniture dataset

# In[28]:


# Source: https://fred.stlouisfed.org/series/RSFHFSN
# Advance Retail Sales: Furniture and Home Furnishings Stores
# Units are in Millions of Dollars, not seasonally adjusted, prices
# Date period: 01.01.1992 to 01.09.2020


# In[29]:


furniture = pd.read_csv('furniture.csv')
df_f = pd.DataFrame(furniture)

col_names = ['month','millions_of_dollars']
df_f.columns = col_names

df_f.head()


# In[30]:


# Always check for null values
df_f.isna().sum()


# In[31]:


df_f.month = pd.to_datetime(df_f.month)

df_f.set_index('month', inplace = True)


# In[32]:


df_f.head(2)


# In[33]:


df_f.describe()


# Notice that the mean is 7624.5 and the median is 7685.0. The max value is 11297.0 and the min value is 3846.0.

# In[34]:


# Plot

plt.style.use('Solarize_Light2')

ax = df_f.plot(color = 'blue', grid = False, figsize=(10,5))

ax.set_xlabel('Year') # Notice the use of set_ to begin methods
ax.set_ylabel('Millions of Dollars')
ax.set_title('Retail sales of Furniture and Home Furnishings Stores')

# Add brown vertical line
ax.axvline('2001-03-01', color = 'brown', linestyle = '--')
ax.axvline('2001-10-01', color = 'brown', linestyle = '--')

ax.axvline('2007-12-01', color = 'brown', linestyle = '--')
ax.axvline('2009-06-01', color = 'brown', linestyle = '--')

ax.axvline('2020-01-01', color = 'brown', linestyle = '--')
ax.axvline('2020-08-01', color = 'brown', linestyle = '--')


# These lines show the periods where there was a change in trend, and a quick look tells us these are the recession periods

# In[35]:


# Plot histogram (frequency of counts), change number of bins to see different plots

df_f.plot(kind = 'hist', bins = 40, color = 'green', grid = True)

# Frequency count of column 'millions_of_dollars'
# count = df_f['millions_of_dollars'].value_counts()
# print(count)


# In[36]:


# Calculate KDP
# A density plot shows distribution of data over a continuous interval.
# KDP smoothes out the noise in time series data.
# The peaks of the density plot help display where values are concentrated over the interval.
# Kernel Density Plot is a better way to display the distribution because it's not affected by the number of bins 
# used (each bar used in a typical histogram).

df_f.plot(kind = 'kde', color = 'purple', grid = False)


# ##### Price Adjustment

# In[37]:


# Source: https://fred.stlouisfed.org/series/CPIAUCSL
# Consumer Price Index: All Items in US City Average, All Urban Consumers (CPIAUCSL)
# Index 1982-1984 = 100, Seasonally Adjusted
# Date period: 01.01.1992 to 01.09.2020, monthly
# Unit is millions of dollards


# In[38]:


cpi = pd.read_csv('cpi_index.csv')

df_cpi = pd.DataFrame(cpi)
col_names = ['date','cpi']
df_cpi.columns = col_names
df_cpi.head()


# In[39]:


df_cpi.tail()


# In[40]:


df_cpi.drop(df_cpi.tail(1).index,inplace=True)

cpi_list = df_cpi['cpi'].to_list()


# In[41]:


df_f['cpi'] = cpi_list
df_f.head()


# In[42]:


# Store the last CPI value
sept2020_cpi = 260.209
sept2020_cpi


# In[43]:


# Calculate the CPI for all months from 1992 to 2020 by dividing by the Sept 2020 CPI value

df_f['cpi_sept20_rate'] = df_f.cpi/sept2020_cpi
df_f.head()


# In[44]:


# Calculate the furniture sales (millions of dollars) in terms of Spet 2020 dollars

df_f['furniture_price_adjusted'] = df_f['millions_of_dollars'] * df_f['cpi_sept20_rate']
df_f.head()


# In[45]:


# Create a new Dataframe with the column that we want
df_f_adj = df_f[['furniture_price_adjusted']]
df_f_adj.head()


# #### Example 3: Bank of America dataset

# In[46]:


bac = data.DataReader("BAC",
                         start = '1990-1-1',
                         end = '2019-10-15',
                         data_source='yahoo')['Adj Close']

df_bac = pd.DataFrame(bac)

df_bac.reset_index(inplace = True)
df_bac['Date'] = pd.to_datetime(df_bac.Date)

df_bac.set_index('Date', inplace = True)
df_bac.head()


# In[47]:


df_bac.plot(title = 'BOA Adj. Closing Price', figsize = (10,5))


# In[48]:


df_bac.isna().sum()


# In[49]:


df_bac.describe()


# In[50]:


print(plt.style.available)


# In[51]:


plt.style.use('_classic_test_patch')

ax = df_bac.plot(color = 'red', grid = True, figsize = (12,5), linewidth = 3)

ax.set_xlabel('Year')
ax.set_ylabel('Adjsuted Close Price')
ax.set_title('Bank of America Adjusted Close Price')

plt.show()


# In[52]:


df_bac.plot(kind = 'hist', bins = 50, color = 'violet', grid = True)


# In[53]:


df_bac.plot(kind = 'kde', color = 'red', grid = True, linewidth = 3, fontsize = 10)


# #### Example 4: J.P. Morgan dataset

# In[54]:


jpm = data.DataReader("JPM",
                         start = '1990-1-1',
                         end = '2019-10-15',
                         data_source='yahoo')['Adj Close']

df_jpm = pd.DataFrame(jpm)

df_jpm.reset_index(inplace = True)
df_jpm['Date'] = pd.to_datetime(df_jpm.Date)

df_jpm.set_index('Date', inplace = True)
df_jpm.head()


# In[55]:


df_jpm.plot(title = 'JP Morgan Adjusted Closing Price', figsize = (10,5))


# In[56]:


df_jpm.isna().sum()


# In[57]:


df_jpm.describe()


# In[58]:


plt.style.use('tableau-colorblind10')

ax = df_jpm.plot(color = 'blue', grid = True, figsize = (12,5), linewidth = 3)

ax.set_xlabel('Year')
ax.set_ylabel('Adjusted Close Price')
ax.set_title('JP Morgan Adjusted Close Price')

plt.show()


# In[59]:


df_jpm.plot(kind = 'hist', bins = 40, color = 'brown', grid = True)


# In[60]:


df_jpm.plot(kind = 'density', color = 'green', grid = True)


# #### Example 5: Average Temperature dataset

# In[61]:


# Source: national Centerse for Environmental Information, National Oceanic and Atmospheric Administraion
# Source: https://www.ncdc.noaa.gov/cag/city/time-series/USW00013994/tavg/all/1/1930-2019?
# Average temperatures, all months, Saint Louis, Missouri, 1938-04 to 2019-01
# Anomaly: Departure from mean relative to the month (1938-2000) base period, Missing value is -99.0


# In[62]:


temp = pd.read_csv('temperatures.csv', skiprows = 4, infer_datetime_format = True)

df_t = pd.DataFrame(temp)
col_names = ['date','avg_temp','anomaly']
df_t.columns = col_names

df_t.head()


# In[63]:


idx_pos = df_t.query('avg_temp == -99.0')
idx_pos


# In[64]:


df_t['avg_temp'].loc[898,]


# In[65]:


df_t['avg_temp'].loc[900,]


# In[66]:


new_avg_val = (df_t['avg_temp'].loc[898,] + df_t['avg_temp'].loc[900,]) / 2
new_avg_val


# In[67]:


# let's first put NaN isntead of -99.0
df_t.at[899,'avg_temp'] = np.nan
df_t['avg_temp'].loc[899,]


# In[68]:


df_t.isna().sum()


# In[69]:


# Lets use interpolation method to put a value in the NaN's place
df_t = df_t.interpolate(method = 'linear', limit_direction = 'forward')

# Check the value where the previous NaN originally coded as -99.0 was at
df_t['avg_temp'].loc[899,]


# In[70]:


df_t.head()


# In[71]:


df_t['date'] = pd.to_datetime(df_t['date'], format = '%Y%m')


# In[72]:


df_t.head()


# In[73]:


df_t.describe()


# In[74]:


# Subset the column of interest
df_at = df_t[['avg_temp']]
df_at.head()


# In[75]:


# Plot time series for the column of interest

plt.style.use('fivethirtyeight')

ax = df_at.plot(color = 'blue', grid = True, figsize = (10,5), linewidth = 1)

ax.set_xlabel('Monthly')
ax.set_ylabel('Temperatures')
ax.set_title('Monthly Average of St. Louis in Fahrenheit')

plt.show()


# In[76]:


df_at.plot(kind = 'hist', bins = 55, color = 'orange', grid = True)


# In[77]:


df_at.plot(kind = 'density', color = 'red', grid = True, linewidth = 3)


# ### Modelling and Decomposing Time Series Based on Trend and Seasonality

# We will focus specifically on:
# 1. Components of a time series. We'll talk about trend, seasonality, and noise
# 2. Modelling time series (additive and multiplicative) 
# 3. Decomposing time series

# Components of a Time Series:
# A time series is composed of mainly trend, seasonality and noise. We will take a look at the component parts of time series, focusing on automated decomposition methods. This will give us some intuition about the components of time series.
# 
# *Trend*
# When we talk about trend, we are talking about how the series data increases or decreases over time. Is it moving higher or lower over the time frame? The series is either uptrend or downtrend, both of which are non-stationary.
# 
# *Seasonality*
# Seasonality refers to a repeating periodic or cyclical pattern with regular itnervals within series. The pattern is within a fixed time period and it repeats itself at regular intervals. There can be upward or downward swings but it continues to repeat over a fixed period of time as in cycle. Cycicality could repeat but it has no fixed point.
# 
# *Noise*
# In general, noise captures the irregularities or random variation in the series. It can have erratic events or simple random variation. It has a short duration. It is hard to predict due to its erratic occurrence.

# *Additive Model*: An additive model is linear. y(t) = Level + Trend + Seasonality + Noise. 
# It is useful when the variations around the trend do not vary with the level of the time series. Components are added together.
# 
# 
# *Multiplicative Model*: A multiplicative model is a non-linear: y(t) = Level * Trend * Seasonality * Noise. 
# A non-linear seasonality has an increasing or decreasing frequency and/or amplitude over time. It is useful when the trend is proportional to the level of the time series. Components are multiplied together.

# #### Decomposing Time Series

# ##### Application of Addictive Model

# In[78]:


# Example 1: Google trends, numbers of searches of the word "vacation", time series shows seasonality 

from statsmodels.tsa.seasonal import seasonal_decompose

ts1 = df_v['num_search_vacation']
# freq is the number of data points in a repeated cycle
result = seasonal_decompose(ts1, model = 'additive', period = 12)

print(result.trend)
print(result.seasonal)
print(result.resid) 
print(result.observed)


# In[79]:


# Example 1: Google trends, numbers of searches of the word "vacation", time series shows seasonality 
from statsmodels.tsa.seasonal import seasonal_decompose
from pylab import rcParams

rcParams['figure.figsize'] = 6,8
plt.style.use('Solarize_Light2')

ts1 = df_v['num_search_vacation']
# freq is the number of data points in a repeated cycle
result = seasonal_decompose(ts1, model = 'additive', period = 12)
result.plot()
plt.tight_layout()
plt.show()


# In[80]:


# Example 2: Furniture sales in Millions of Dollars (adjusted to July 2019 prices)
from statsmodels.tsa.seasonal import seasonal_decompose
from pylab import rcParams

rcParams['figure.figsize'] = 6,8
plt.style.use('Solarize_Light2')

ts2 = df_f['furniture_price_adjusted']
# freq is the number of data points in a repeated cycle
result = seasonal_decompose(ts2, model = 'additive', period = 12)
result.plot()
plt.tight_layout()
plt.show()


# In[81]:


# Example 3: Adjusted Close Price of Bank of America
from statsmodels.tsa.seasonal import seasonal_decompose
from pylab import rcParams

rcParams['figure.figsize'] = 6,8
plt.style.use('Solarize_Light2')

ts3 = df_bac['Adj Close']
# freq is the number of data points in a repeated cycle
result = seasonal_decompose(ts3, model = 'additive', period = 365)
result.plot()
plt.tight_layout()
plt.show()


# In[82]:


# Example 4: Adjusted Closed Price for JP Morgan
from statsmodels.tsa.seasonal import seasonal_decompose
from pylab import rcParams

rcParams['figure.figsize'] = 6,8
plt.style.use('bmh')

ts4 = df_jpm['Adj Close']
# freq is the number of data points in a repeated cycle
result = seasonal_decompose(ts4, model = 'additive', period = 365)
result.plot()
plt.tight_layout()
plt.show()


# In[83]:


# Example 5: Average Temperature of St. Louis
from statsmodels.tsa.seasonal import seasonal_decompose
from pylab import rcParams

rcParams['figure.figsize'] = 6,8
plt.style.use('Solarize_Light2')

ts5 = df_t['avg_temp']
# freq is the number of data points in a repeated cycle
result = seasonal_decompose(ts5, model = 'additive', period = 12)
result.plot()
plt.tight_layout()
plt.show()


# ##### Application of Multiplicative Model

# In[84]:


# Example 1: Google trends, numbers of searches of the word "vacation", time series shows seasonality 

from statsmodels.tsa.seasonal import seasonal_decompose

ts1 = df_v['num_search_vacation']
# freq is the number of data points in a repeated cycle
result = seasonal_decompose(ts1, model = 'multiplicative', period = 12)

print(result.trend)
print(result.seasonal)
print(result.resid) 
print(result.observed)


# In[85]:


# Example 1: Google trends, numbers of searches of the word "vacation", time series shows seasonality 
from statsmodels.tsa.seasonal import seasonal_decompose
from pylab import rcParams

rcParams['figure.figsize'] = 6,8
plt.style.use('bmh')

ts1 = df_v['num_search_vacation']
# freq is the number of data points in a repeated cycle
result = seasonal_decompose(ts1, model = 'multiplicative', period = 12)
result.plot()
plt.tight_layout()
plt.show()


# In[86]:


# Example 2: Furniture sales in Millions of Dollars (adjusted to July 2019 prices)
from statsmodels.tsa.seasonal import seasonal_decompose
from pylab import rcParams

rcParams['figure.figsize'] = 6,8
plt.style.use('Solarize_Light2')

ts2 = df_f['furniture_price_adjusted']
# freq is the number of data points in a repeated cycle
result = seasonal_decompose(ts2, model = 'multiplicative', period = 12)
result.plot()
plt.tight_layout()
plt.show()


# In[87]:


# Example 3: Adjusted Close Price of Bank of America
from statsmodels.tsa.seasonal import seasonal_decompose
from pylab import rcParams

rcParams['figure.figsize'] = 6,8
plt.style.use('Solarize_Light2')

ts3 = df_bac['Adj Close']
# freq is the number of data points in a repeated cycle
result = seasonal_decompose(ts3, model = 'multiplicative', period = 365)
result.plot()
plt.tight_layout()
plt.show()


# In[88]:


# Example 4: Adjusted Closed Price for JP Morgan
from statsmodels.tsa.seasonal import seasonal_decompose
from pylab import rcParams

rcParams['figure.figsize'] = 6,8
plt.style.use('bmh')

ts4 = df_jpm['Adj Close']
# freq is the number of data points in a repeated cycle
result = seasonal_decompose(ts4, model = 'multiplicative', period = 365)
result.plot()
plt.tight_layout()
plt.show()


# In[89]:


# Example 5: Average Temperature of St. Louis
from statsmodels.tsa.seasonal import seasonal_decompose
from pylab import rcParams

rcParams['figure.figsize'] = 6,8
plt.style.use('Solarize_Light2')

ts5 = df_t['avg_temp']
# freq is the number of data points in a repeated cycle
result = seasonal_decompose(ts5, model = 'multiplicative', period = 12)
result.plot()
plt.tight_layout()
plt.show()


# ### Approaches to Detrend and Deseasonalize a Time Series

# 1. Differencing (first difference, second difference, and substraction from the mean value)
# 2. Change in Percentage - typically called return with stock price data
# 3. Transformation (log transformation)- if you have a range of data that takes on many different values and the values themselves are big, a log transformation might be a good idea

# When detrending and/or deseasonising a time series you may use one or a combination of approaches such as differencing, subtracting from the mean (or from a mean over a period), calculating percentage change, or using a transformation such as a log transformation. For instance, you make the first difference and then a log transformation or you may use a second difference. As you perform various operations (first dfference, second difference, subtracting the mean, log transformation, percentage), you will gain some intuition about how your data is being altered.
# 
# It is not always necessary to detrend and deseasonalise. Be aware that this may or may not apply to your particular data or domain. At times, you can work with the original data as it is.

# #### Differencing
# 
# Differencing can help to reduce trend and seasonality. You can difference your data by substraction. You subtract away the previous period from the current period. Below you will see that the first observation has a NaN because there was no previous period.

# In[90]:


# Google trends pn a search term "vacation" from 2004 to 2020
df_v.head()


# In[91]:


# Example of first differencing
df_v['first_diff'] = df_v['num_search_vacation'].diff()

df_v.head()


# In[92]:


# Drop the NaN values

df_v.dropna(inplace = True)


# In[93]:


df_v.describe()


# In[94]:


# Example of second differencing

df_v['second_diff'] = df_v['num_search_vacation'].diff(2)
df_v.dropna(inplace = True)
df_v.describe()


# Comparing the mean and std deviation between the first and second difference, the first difference gives better results and should be sufficient to detrend and deseasonilise the series.

# In[95]:


# Plot the time series

plt.style.use('bmh')

ax = df_v['first_diff'].plot(color = 'green', grid = True, figsize = (12,5), linewidth = 2)

ax.set_xlabel('Year')
ax.set_ylabel('Number of searches for the term "vacation"')
ax.set_title('Google Trends of Vacation')

plt.show()


# In[96]:


# Plot Kernel Density Plot

df_v['first_diff'].plot(kind = 'kde')


# ##### Examples of Subtraction from Mean
# 
# You can difference your data by subtraction. You subtract away the previous period from the current period. Below you will see that the first observation has an NaN because there was no previous period.

# In[97]:


# Average Temperature of St. Louis
df_t.head()


# In[98]:


df_t['mean_diff'] = df_t['avg_temp'] - df_t['avg_temp'].mean()
df_t.head()


# In[99]:


df_t.describe()


# In[100]:


# Plot the series
plt.style.use('bmh')

ax = df_t['mean_diff'].plot(color = 'green', grid = True, figsize = (10,5))
ax.set_xlabel('Year')
ax.set_ylabel('Difference (anomaly) from the mean')
ax.set_title('Avg Temperature of St. Louis')

plt.show()


# In[101]:


df_t['mean_diff'].plot(kind = 'density', grid = True, linewidth = 2)


# #### Percentage Change

# ##### Furniture example

# In[102]:


# Furniture sales in Millions of Dollars (adjusted to September 2020 prices)

df_f.head()


# In[103]:


df_f['furniture_pct_change'] = df_f['furniture_price_adjusted'].pct_change()
df_f.head()


# In[104]:


df_f.dropna(inplace = True)


# In[105]:


df_f.describe()


# In[106]:


# Plot the time series

plt.style.use('bmh')

ax = df_f['furniture_pct_change'].plot(color = 'violet', grid = True, figsize = (12,5), linewidth = 2)

ax.set_xlabel('Year')
ax.set_ylabel('Percentage Change in Millions of Dollars')
ax.set_title('Retail Sales of Furniture and Furnishings')

plt.show()


# In[107]:


df_f['furniture_pct_change'].plot(kind = 'kde')


# ##### Bank of America example

# In[108]:


# Another example of pct change for adjusted close stock price
# Bank of America example

df_bac.head()


# In[109]:


# Convert daily data to weekly data

df_bac = df_bac.resample(rule = 'W').last()

# rule is weekly
# last means the last day of the week


# In[110]:


df_bac['pct_change'] = df_bac['Adj Close'].pct_change()


# In[111]:


df_bac.dropna(inplace = True)
df_bac.head()


# In[112]:


df_bac.describe()


# In[113]:


# Plot the graph
plt.style.use('bmh')

ax = df_bac['pct_change'].plot(color = 'pink', grid = True, figsize = (10,5))
ax.set_xlabel('Year')
ax.set_ylabel('Percentage change in Adjusted Closed Price')
ax.set_title('Bank of America')


# In[114]:


df_bac['pct_change'].plot(kind = 'density', grid = True, linewidth = 2)


# ##### JP Morgan

# In[115]:


df_jpm.head()


# In[116]:


df_jpm = df_jpm.resample(rule = 'W').last()

df_jpm['pct_change'] = df_jpm['Adj Close'].pct_change()
df_jpm.dropna(inplace = True)
df_jpm.head()


# In[117]:


df_jpm.describe()


# In[118]:


plt.style.use('bmh')

ax = df_jpm['pct_change'].plot(color = 'blue', grid = True, figsize = (10,5))
ax.set_xlabel('Year')
ax.set_ylabel('Percentage Change in Price in US Dollars')
ax.set_title('Adjusted Closing Price for JP Morgan')

plt.show()


# In[119]:


df_jpm['pct_change'].plot(kind = 'kde', grid = True, linewidth = 2)


# #### Log Transformation

# In[120]:


df_f['furniture_log'] = np.log(df_f['furniture_price_adjusted'])
df_f.head()


# In[121]:


df_f.describe()


# In[122]:


df_f['furniture_log'].plot()


# In[123]:


df_f['furniture_log'].plot(kind = 'kde')


# ### Correlation: Relationship between series

# Types:
# 
# 1. Low Correlation
# 2. Medium Correlation
# 3. High Correlation

# Correlation is a measure of the direction and strength of a relationship. In this case, we are looking at two time series. Measurements are from -1 to 1 with 0 meaning no relationship or zero correlation between the two series. 1 emans a perfect, positive correlation, and -1 means a perfect, negative correlation. High correlations are toward 1 or -1 whereas low correlations are toward 0. In time series, one must be cautioned when viewing correlation, given the time period. You can imagine that on one day, there is strong, positive correlation but that the very next day, it could be a different story about the relationship between the two series.

# #### Low Correlation

# ##### Google trends: Freedom and Choice

# In[124]:


freedom_choice = pd.read_csv('freedom_choice.csv', skiprows = 1)

col_names = ['week','freedom','choice']
freedom_choice.columns = col_names

df_fc = pd.DataFrame(freedom_choice)
df_fc['week'] = pd.to_datetime(df_fc['week'], infer_datetime_format = True)

df_fc.set_index('week', inplace = True)

df_fc.head()


# In[125]:


df_fc.isna().sum()


# In[126]:


df_fc.describe()


# In[127]:


df_fc.plot(figsize = (8,5), subplots = True)


# In[128]:


# Perform correlation

df_fc['freedom'].corr(df_fc['choice'])


# ##### Google Trends: Globalism and Localism

# In[129]:


glocal = pd.read_csv('global_local.csv', skiprows = 2)

col_names = ['week','globalism','localism']
glocal.columns = col_names

df_gc = pd.DataFrame(glocal)
df_gc['week'] = pd.to_datetime(df_gc['week'], infer_datetime_format = True)
df_gc.set_index('week', inplace = True)

df_gc.head()


# In[130]:


df_gc.isna().sum()


# In[131]:


df_gc.describe()


# In[132]:


df_gc.plot(figsize = (12,5), subplots = True)


# In[133]:


df_gc['globalism'].corr(df_gc['localism'])


# #### Medium Correlation

# ##### Adjusted stock price of Bank of America and JP Morgan

# In[134]:


df_bac.head()


# In[135]:


df_jpm.head()


# In[136]:


df_banks = pd.concat([df_bac,df_jpm], axis = 1)
df_banks.head()


# In[137]:


col_names = ['bac_closeprice','bac_pct_change','jpm_closeprice','jpm_pct_change']
df_banks.columns = col_names

df_banks.head()


# In[138]:


df_banks[['bac_closeprice','jpm_closeprice']].plot()


# In[139]:


df_banks['bac_closeprice'].corr(df_banks['jpm_closeprice'])


# ###### Rolling window of Correlation

# In[140]:


df_banks['bac_closeprice'].rolling(window = 6).corr(df_banks['jpm_closeprice']).head(50)


# In[141]:


# Calculate rolling window of correlation
df_banks['bac_closeprice'].rolling(window = 6).corr(df_banks['jpm_closeprice']).tail(50)


# ##### Google Trends: Materialism and Consumerism

# In[142]:


mat_con = pd.read_csv('materialism_consumerism.csv', skiprows = 2)

col_names = ['month','materialism','consumerism']
mat_con.columns = col_names

df_mc = pd.DataFrame(mat_con)
df_mc['month'] = pd.to_datetime(df_mc['month'], infer_datetime_format = True)

df_mc.set_index('month', inplace = True)
df_mc.head()


# In[143]:


df_mc.plot(figsize = (12,5))


# In[144]:


df_mc['materialism'].corr(df_mc['consumerism'])


# ##### Google Trends: Des Moines and New York

# In[145]:


des_ny = pd.read_csv('desmon_ny.csv', skiprows = 2)

col_names = ['month','des_moines','new_york']
des_ny.columns = col_names

df_cities = pd.DataFrame(des_ny)
df_cities['month'] = pd.to_datetime(df_cities['month'], infer_datetime_format = True)

df_cities.set_index('month', inplace = True)

df_cities.head()


# In[146]:


df_cities.plot(figsize = (12,5))


# If you eyeball it, you'll probably notice no relation/very low correlation. But let's see what the correlation is according to Python

# In[147]:


df_cities['des_moines'].corr(df_cities['new_york'])


# #### High Correlation

# ##### Google Trends: Growth and Economy

# In[148]:


grow_eco = pd.read_csv('growth_economy.csv', skiprows = 2)

col_names = ['month','growth','economy']
grow_eco.columns = col_names

df_ge = pd.DataFrame(grow_eco)
df_ge['month'] = pd.to_datetime(df_ge['month'], infer_datetime_format = True)

df_ge.set_index('month', inplace = True)

df_ge.head()


# In[149]:


df_ge.describe()


# In[150]:


df_ge.plot(figsize = (12,5))


# In[151]:


df_ge['growth'].corr(df_ge['economy'])


# In[152]:


grow_eco_ly = pd.read_csv('growth_eco_lastyear.csv', skiprows = 1)

col_names = ['week','growth','economy']
grow_eco_ly.columns = col_names

df_ge_year = pd.DataFrame(grow_eco_ly)
df_ge_year['week'] = pd.to_datetime(df_ge_year['week'], infer_datetime_format = True)

df_ge_year.set_index('week', inplace = True)

df_ge_year.head()


# In[153]:


df_ge_year.plot(figsize = (12,5))


# In[154]:


df_ge_year['growth'].corr(df_ge_year['economy'])


# ##### Google Trends: Population and Crime 

# In[155]:


pop_crime = pd.read_csv('population_crime.csv', skiprows = 1)

col_names = ['month','population','crime']
pop_crime.columns = col_names

df_pc = pd.DataFrame(pop_crime)
df_pc['month'] = pd.to_datetime(df_pc['month'], infer_datetime_format = True)

df_pc.set_index('month', inplace = True)

df_pc.head()


# In[156]:


df_pc.describe()


# In[157]:


df_pc.isna().sum()


# In[158]:


df_pc.plot(figsize = (12,5))


# In[159]:


df_pc['population'].corr(df_pc['crime'])


# ### Autocorrelation: Relationship within Series

# Why do we autocorrelate data?
# 
# Autocorrelation plots (Box and Jenkins, pp. 28-32) are a commonly-used tool for _checking randomness in a data set_. This randomness is ascertained by computing autocorrelations for data values at varying time lags. If random, such autocorrelations should be near zero for any and all time-lag separations.

# 1. White Noise
# 2. Autocorrelation function (ACF)
# 3. Partial autocorrelation function (PACF)

# Much like correlation, autocorrelation gives a measure of the strength and direction of a relationship between two time series. Autocorrelation is done with a variable and its lag. It's a correlation with itself, hence autocorrelation. Basically, you are using the same time series and creating a second series, which is shifted by a time step. It is common to use the autocorrelation (ACF) plot to visualise the autocorrelation of a time-series. Let's start by taking a look at white noise:

# #### White Noise, ACF, PACF
# 
# If your time series is statistically uncorrelated (independent) with zero mean and constant variance, then you have a particular type of noise called white noise.

# In[160]:


# loc is mean, scale is standard deviation
white_noise = np.random.normal(loc = 0, scale = 2, size = 1000)

plt.plot(white_noise)


# In[161]:


white_noise.mean(), white_noise.std()


# In[162]:


# Plot autocorrelation
from statsmodels.graphics.tsaplots import plot_acf
# Example of white noise autocorrelation (serial correlation)
plot_acf(white_noise, lags = 100);


# Be aware that all the lags have close to zero autocorrelation so this means that lagged values cannot be used in prediction. All the lagged variables are random and are not useful for prediction.

# ##### Example 1: Vacation Data

# In[163]:


df_v.head()


# In[164]:


# Lag plot
from pandas.plotting import lag_plot

lag_plot(df_v['num_search_vacation'])
plt.show()


# The plot above shows the vacation data on the y(t) against its previous time step, prior month, the y(t+1). You can think of the data shifted by a month and then plotted, removing the first data point. Clearly, we see a positive relationship, though it has a broader range of scatter.

# In[165]:


# Autocorrelation
autocorrelation = df_v['num_search_vacation'].autocorr()
autocorrelation


# The correlation between the vacation data and it lag variable by a time step shows generally medium to high correlation at 0.80. This indicates that there is a substantial, positive relationship.

# In[166]:


# Plot ACF to visualise the autocorrelation
plot_acf(df_v['num_search_vacation'], lags = 12);


# The partial autocorrelation function (PACF) gives the partial correlation of a time series with its own lagged values. It controls for other lags. The idea for the PACF is taht we perform correlation between a variable and itself lagged and then we subtract that effect from the variable and then find out what residual correlation is left over from that variable and further lags. For example, a PACF of order 3 returns the correlation between our time series (t1,t2,t3,..) and its own values lagged by 3 time points (t4,t5,t6,..) but only after removing all effects attributable to lags 1 and 2.

# In[167]:


# Plot Partial autocorrelation function (PACF)

from statsmodels.graphics.tsaplots import plot_pacf

plot_pacf(df_v['num_search_vacation'],lags = 12);


# If partial autocorrelation values are close to 0, then values between observations and lagged observations are not correlated with one another. Inversely, partial autocorrelations with values close to 1 or -1 indicate that there exists strong positive or negative correlations between the lagged observations of the time series.
# 
# 
# The .plot_pacf() function also returns confidence intervals, which are represented as blue shaded regions. If partial autocorrelation values are beyond this confidence interval regions, then you can assume that the observed partial autocorrelation values are statistically significant.

# ##### Example 2: Furniture Data

# In[168]:


df_f.head()


# In[169]:


# Lag Plot

lag_plot(df_f['furniture_price_adjusted'])
plt.show()


# The plot above shows the furniture data aginst its previous time step, prior month. You can think of the data shifted by a month and then plotted, removing the first data point. Clearly, we see a positive relationship.

# In[170]:


# Autocorrelation

df_f['furniture_price_adjusted'].autocorr()


# In[171]:


# Plot ACF
plot_acf(df_f['furniture_price_adjusted'], lags = 24);


# The x-axis shows the number of lags where the y-axis shows the correlation value. Note that correlation measure runs from -1 to 1. The results show positive correlation. The scores all extend beyond the blue shaded region, which denotes statistical significance. For each time period, the measure is of its current time value's with its prior time value. It shows strong positive, autocorrelation up to 24 lags.

# In[172]:


# Partial Autocorrelation Function
plot_pacf(df_f['furniture_price_adjusted'], lags = 24);


# Strong partial autocorrelation at the first two lags. The candlesticks extend beyond the margin of uncertainty for lags 3 and 4 as well as 11 and 12, going i the positive direction. In terms of negative correlation, lag 10, 13 and 18 show statistical significance in terms of negative correlation.

# ##### Example 3: Adjusted Close Stock Price for Bank of America

# In[173]:


df_bac.head()


# In[174]:


# Plot lag
lag_plot(df_bac['Adj Close'])


# In[175]:


# Calculate Autocorrelation
df_bac['Adj Close'].autocorr()


# In[176]:


# Plot Autocorrelation Function
plot_acf(df_bac['Adj Close'], lags = 24);


# In[177]:


# Plot Partial Autocorrelation Function

plot_pacf(df_bac['Adj Close'], lags = 24);


# ##### Example 4: Adjusted Close Stock Price for JP Morgan

# In[178]:


df_jpm.head()


# In[179]:


# Plot lag
lag_plot(df_jpm['Adj Close'])


# In[180]:


# Plot autocorrelation
df_jpm['Adj Close'].autocorr()


# In[181]:


# Plot Autocorrelation Function
plot_acf(df_jpm['Adj Close'], lags = 24);


# In[182]:


# Plot Partial Autocorrelation Function

plot_pacf(df_jpm['Adj Close'], lags = 24);


# ##### Example 5: Monthly Average Temperature of St. Louis

# In[183]:


df_t.set_index('date', inplace = True)
df_t.head()


# In[184]:


# Plot lag
lag_plot(df_t['avg_temp'])
plt.show()


# In[185]:


# Plot Autocorrelation

df_t['avg_temp'].autocorr()


# In[186]:


# Plot Autocorrelation Function
plot_acf(df_t['avg_temp'], lags = 24);


# In[187]:


# Plot Partial Autocorrelation Function
plot_pacf(df_t['avg_temp'], lags = 24);


# ### Operating with Time Series Models: Stationarity in Time Series

# In certain situations, you may need to make stationary your data, but there are certainly no hard and fast rules. You can achieve stationarity when you remove trend and seasonality such that you have constant mean and variance. In particular data domains and situations, your time series data may need to be made stationary before applying any statistical analysis. But, be aware that even "weakly" stationary data can also be acceptable. So of course, you can always work with the original data and then compare those results against the data that has been made stationary. 

# **Augmented Dickey - Fuller Test**
# 
# To model a time series, it can be stationary or weakly stationary. The distribution of the data does not change over time. The series has zero trend, the variance is constant, and the autocorrelation is constant. The augmented Dicket-Fuller Test is a statistical test for non-stationarity. The null hypothesis is that the time seris is non-stationary due to trend.

# In[188]:


from statsmodels.tsa.stattools import adfuller


# The more negative the test-statistic, then it is more likely that the data is stationary. For the p-value, if it is small such that it is below 0.05, then we reject the null hypothesis. This means we assume that the time series must be stationary. For the critical values, if we want a p-value of 0.05 or below, our test statistic needs to be below the corresponding critical value.

# ##### Example 1: Vacation data, first difference

# In[189]:


df_v.head()


# In[190]:


# Run test
vacation_result = adfuller(df_v['first_diff'])

# Print the test statistic
print(vacation_result[0])

# Print the p-value
print(vacation_result[1])

# Print the critical values for the test statistic for 1%, 5% and 10%
print(vacation_result[4])


# The p-value is very very low (2.65 e to the power of negative 7) which is definitely lower than the 0.05 threshold therefore we reject the null and claim that the series is stationary.

# ##### Example 2: Furniture data, percentage change

# In[191]:


df_f.head()


# In[192]:


# Run the test
furniture_result = adfuller(df_f['furniture_pct_change'])

# Print the test statistic
print(furniture_result[0])

# Print the p-value
print(furniture_result[1])

# Print the critical values for the test statistic at 1%, 5% and 10%
print(furniture_result[4])


# ##### Example 3: Bank of America stock price data, percentage change

# In[193]:


df_bac.head()


# In[194]:


# Run the test
bac_result = adfuller(df_bac['pct_change'])

# Print the test statistic
print(bac_result[0])

# Print the p-value
print(bac_result[1])

# Print the critical values for the test statistic at 1%, 5% and 10%
print(bac_result[4])


# ##### Example 4: JP Morgan stock price data, percentage change

# In[195]:


df_jpm.head()


# In[196]:


# Run the test
jpm_result = adfuller(df_jpm['pct_change'])

# Print the test statistics
print(jpm_result[0])

# print the p-value
print(jpm_result[1])

# Print the critical values for test statistic at 1%, 5% and 10%
print(jpm_result[4])


# ##### Average Temperature of St. Louis, difference from the mean.

# In[197]:


df_t.head()


# In[198]:


# Run the test
temp_result = adfuller(df_t['mean_diff'])

# Print the test statistic
print(temp_result[0])

# Print the p-value
print(temp_result[1])

# Print the critical values for the test statistic at 1%, 5% and 10%
print(temp_result[4])


# ### Autoregression (AR) and Moving Average (MA) Models

# **Autogression model (AR)**: It is a rergession between a time series and itself lagged by a time step or steps
# 
# **Moving Average model (MA)**: It is a regression between a time series and its own residuals lagged by a time step or steps
# 
# **ARMA model**: It is a combination of AR and MA.
# 
# 
# **AR and MA Models**:
# 
# In time series, autoregression(AR) and moving average(MA) models provide a simple description of a process in terms of two factors - one for autoregression and the other for moving average. The AR part entails regressing the variable on its own lagged values. The idea is that the previous time period's value can help us predict the current time period's value. The MA part involves modelling the error term as a linear combination of error terms occuring at the same time step and at vaious times in the past.
# 
# **Autoregression Models**
# 
# Autogressive (AR) model is when present value of a time series can be predicted from using previous values of the same series. It's a regression using its same series, though shifted by a time step, called a lag. The present value is a weighted average of its past values. Both the t-1 and t-2 are lags of the time series y. The error terms (noise) is represented as e. The values a1 and a2 are the coefficients of the model.
# 
# AR(1) model: $y_t$ = $a_1$$y_(t-1)$ + $e_t$
# 
# AR(2) model: $y_t$ = $a_1$$y_(t-1)$ + $a_2$$y_(t-2)$ + $e_t$
# 
# We typically represent the order of the model as p for an AR model such as AR(p).
# 
# 
# **Moving Average Models**
# 
# Moving average (MA) is a process where the present value of a time series, y, is defined as a linear combination of past errors. The error term (noise) is represetned as e. Both the t-1 and t-2 are lags of the time on the errors.
# 
# MA(1) Model: $y_t$ = $m_1$$e_(t-1)$ + $e_t$
# 
# MA(2) Model: $y_t$ = $m_1$$e_(t-1)$ + $m_2$$e_(t-2)$ + $e_t$
# 
# We typically represent the order of the model as q for an MA model such as MA(q).
# 
# 
# **ARMA Models**
# 
# We can combine both Autoregressive and Moving Average models together to create an ARMA model. The time series is regressed on the previous values and the previous errors.
# 
# ARMA(1,1) model = $y_t$ = $a_1$$y_(t-1)$ + $m_1$$e_(t-1)$
# 
# For ARMA(p,q), p is the order of the AR process and q is the order of the MA process.

# #### Examples:

# In[199]:


# This is example code
from statsmodels.tsa.arima_model import ARMA

# model = ARMA(timeseries, order = (p,q)) # General model

# ar_model = ARMA(timeseries, order = (2,0)) # AR model

# ma_model = ARMA(timeseries, order = (0,2)) # MA model

# model = ARMA(timeseries, order = (3,1)) # This means that the AR has 3 lags and MA has 1 lag.

# results = model.fit() # fit the model
# print(results.summary) # print out the summary results


# In[200]:


# simulatione example

# https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima_process.arma_generate_sample.html

# Import data generation function and set random seed
from statsmodels.tsa.arima_process import arma_generate_sample
np.random.seed(41)

# Set ar_coefs and ma_coefs for an MA(1) model with MA lag -1 coefficient of -0.8
ar_coefs = [1] # Remember to set the lag-0 coefficients to 1.
ma_coefs = [1, -0.8] # Remember to set the lag-0 coefficients to 1.

# Generate data, scale is the standard deviation of the errors
data = arma_generate_sample(ar_coefs, ma_coefs, nsample = 100, scale = 0.5)

plt.plot(data)
plt.ylabel(r'$y_t$')
plt.xlabel(r'$t$')
plt.show()


# In[201]:


# AR(2) model is just an ARMA(2,0) model

# Import data generation function and set random seed
from statsmodels.tsa.arima_process import arma_generate_sample
np.random.seed(42)

# Set ar_coefs and ma_coefs for an AR(2) model with AR lag-1 and lag-2 coefficients of 0.3 and 0.2 respectively
ar_coefs = [1, -0.3, -0.2] # Remember to flip the sign of the coefficients.
ma_coefs = [1] # Remember to set the lag-0 coefficients to 1.

# Generate data, scale is the standard deviation of the errors
data = arma_generate_sample(ar_coefs, ma_coefs, nsample = 100, scale = 0.5)

plt.plot(data)
plt.ylabel(r'$y_t$')
plt.xlabel(r'$t$')
plt.show()


# In[202]:


# Set the coefficients for a model with from yt = -0.3yt-1 + 0.2et-1 + 0.4et-2 + et.

# Import data generation function and set random seed.
from statsmodels.tsa.arima_process import arma_generate_sample
np.random.seed(43)

ar_coefs = [1, 0.3] # Remember for lags greater than 0, we need to flip the sign
ma_coefs = [1, 0.2, 0.4] # here we don't flip since these are coefficients of errors

# generate data
data = arma_generate_sample(ar_coefs, ma_coefs, nsample = 100, scale = 0.5)

plt.plot(data)
plt.ylabel(r'$y_t$')
plt.xlabel(r'$t$')
plt.show()


# ### Estimating an AR Model

# An autoregression model is a regression with a time series and itself, shifted by a time step or steps. These are called lags. I will demonstrate with five examples with the non-stationarized datasets so that you can see the results in the original dataset along with the forecasted dataset.

# #### Example 1: Vacation dataset

# In[203]:


from statsmodels.graphics.tsaplots import plot_acf
from pandas.plotting import lag_plot

df_v.head()


# In[204]:


# Plot the time series against its lag
lag_plot(df_v['num_search_vacation'])
plt.show()


# In[205]:


# Find the correlation of the time series with its lag
values = pd.DataFrame(df_v['num_search_vacation'].values)
dataframe = pd.concat([values.shift(1),values], axis = 1)
dataframe.columns = ['t-1', 't+1']
result = dataframe.corr()
print(result)


# In[206]:


# Plot the autocorrelation of the dataset
from pandas.plotting import autocorrelation_plot

autocorrelation_plot(df_v['num_search_vacation'])
plt.show()


# In[207]:


# Plot the autocorrelation function
from statsmodels.graphics.tsaplots import plot_acf

plot_acf(df_v['num_search_vacation'], lags = 50);
plt.show()


# In[208]:


# Estimate the AR(1) model

# import the ARMA module from statsmodals
from statsmodels.tsa.arima_model import ARMA

# Fit an AR(1) model to the first simulated data
mod = ARMA(df_v['num_search_vacation'], order = (1,0)) # fit data to AR(1) model
res = mod.fit()

# Print result summary
print(res.summary())
# Print result parameters - are they close to true parameters?
print(res.params)


# In[209]:


# Forecasting

res.plot_predict(start = '2015', end = '2025')
plt.show()


# #### Example 2: Furniture dataset

# In[210]:


df_f.head()


# In[211]:


# Plot the timeseries against its lag
from pandas.plotting import lag_plot

lag_plot(df_f['furniture_price_adjusted'])


# In[212]:


# Find the correlation of the time series with its lag

values = pd.DataFrame(df_f['furniture_price_adjusted'].values)
dataframe = pd.concat([values.shift(1), values], axis = 1)
dataframe.columns = ['t-1', 't+1']
result = dataframe.corr()
print(result)


# In[213]:


# Calculate the autocorrelation of the time series with its lag
from pandas.plotting import autocorrelation_plot

autocorrelation_plot(df_f['furniture_price_adjusted'])
plt.show()


# In[214]:


# Plot the autocorrelation function
from statsmodels.graphics.tsaplots import plot_acf

plot_acf(df_f['furniture_price_adjusted'], lags = 50);
plt.show()


# In[215]:


# Estimate the AR(1) model
from statsmodels.tsa.arima_model import ARMA

mod = ARMA(df_f['furniture_price_adjusted'], order = (1,0))
res = mod.fit()

# print the result summary
print(res.summary())

# print the result parameter - are they similar to the original values?
print(res.params)


# In[216]:


# Forecast

res.plot_predict(start = '2015', end = '2025')


# #### Example 3: Bank of America dataset

# In[217]:


df_bac = df_bac.resample(rule = 'M').last()

df_bac.head()


# In[218]:


# plot the lag
from pandas.plotting import lag_plot

lag_plot(df_bac['Adj Close'])
plt.show()


# In[219]:


# Calculate the correlation between the time series and its lag

values = pd.DataFrame(df_bac['Adj Close'].values)
dataframe = pd.concat([values.shift(1),values], axis = 1)
dataframe.columns = ['t-1','t+1']
result = dataframe.corr()
print(result)


# In[220]:


# Calculate the autocorrelation
from pandas.plotting import autocorrelation_plot

autocorrelation_plot(df_bac['Adj Close'])
plt.show()


# In[221]:


# Plot the autocorrelation function
from statsmodels.graphics.tsaplots import plot_acf

plot_acf(df_bac['Adj Close'], lags = 50);
plt.show()


# In[222]:


# Estimate the AR(1) model
from statsmodels.tsa.arima_model import ARMA

mod = ARMA(df_bac['Adj Close'], order = (1,0))
res = mod.fit()

# print summary of result
print(res.summary())

# print parameters of the result
print(res.params)


# In[223]:


# Forecast

res.plot_predict(start = '2015', end = '2025')


# ### Estimating an MA Model

# Moving Average model was described as a regression between a time series and its lagged error term. By specifying an order, we set the number of lags.

# #### Example 1: Vacation dataset

# In[224]:


df_v.head()


# In[225]:


# Estimate moving average

# Import ARMA
from statsmodels.tsa.arima_model import ARMA

mod = ARMA(df_v['num_search_vacation'], order = (0,1))
res = mod.fit()

print(res.summary())
print(res.params)


# In[226]:


# Forecast

res.plot_predict(start = '2015', end = '2025')
plt.show()


# #### Example 2: JP Morgan dataset

# In[227]:


# convert from weekly to quartly
df_jpm = df_jpm.resample(rule = 'Q').last()
df_jpm.head()


# In[228]:


# Estimate
from statsmodels.tsa.arima_model import ARMA

mod = ARMA(df_jpm['Adj Close'], order = (0,1))
res = mod.fit()

print(res.summary())

print(res.params)


# In[229]:


# Forecast

res.plot_predict(start = '2015', end = '2025');
plt.show()


# #### Example 3: Temperature dataset

# In[230]:


df_t.head()


# In[231]:


# Estimate the moving average

mod = ARMA(df_t['avg_temp'], order = (0,1))
res = mod.fit()

print(res.summary())
print(res.params)


# In[232]:


# Forecast the MA
res.plot_predict(start = '2015', end = '2025')


# ### Building an ARMA model

# We will put autoregression and moving average models together in combination. AR(p) models try to explain the momentum and mean reversion effects. MA(q) models try to capture the shock effects observed in the white noise terms. These effects are unexpected or surprise events, hence anomalies.

# #### Example 1: Vacation dataset

# In[233]:


df_v.head()


# In[234]:


mod = ARMA(df_v['num_search_vacation'], order = (1,1))
res = mod.fit()

print(res.summary())

print(res.params)


# In[235]:


res.plot_predict(start = '2015', end = '2025');
plt.show()


# #### Example 2: Furniture dataset

# In[236]:


df_f.head()


# In[237]:


mod = ARMA(df_f['furniture_price_adjusted'], order = (1,1))
res = mod.fit()

print(res.summary())
print(res.params)


# In[238]:


res.plot_predict(start = '2015', end = '2025');
plt.show()


# #### Example 3: Bank of America dataset

# In[239]:


df_bac.head()


# In[240]:


mod = ARMA(df_bac['Adj Close'], order = (1,1))
res = mod.fit()

print(res.summary())
print(res.params)


# Here the `ma.L1.Adj Close` has a p value of 0.832 which is much higher than the threshold of 0.05. Therefore we fail to reject the null here.

# In[241]:


res.plot_predict(start = '2015', end = '2025');
plt.show()


# #### Example 4: JP Morgan dataset

# In[242]:


df_jpm.head()


# mod = ARMA(df_jpm['Adj Close'], order = (1,1))
# 
# res = mod.fit()
# 
# print(res.summary())
# 
# print(res.params)
# 
# _on running the above code we get the following error:_
# ValueError: The computed initial AR coefficients are not stationary
# You should induce stationarity, choose a different model order, or you can
# pass your own start_params.

# In[243]:


mod = ARMA(df_jpm['pct_change'], order = (1,1))

res = mod.fit()

print(res.summary())

print(res.params)


# In[244]:


res.plot_predict(start = '2015', end = '2025');
plt.show()


# This is a terrible forecast for the ARMA model

# #### Example 5: Temperature dataset

# In[245]:


df_t.head()


# In[246]:


mod = ARMA(df_t['avg_temp'], order = (1,1))
res = mod.fit()

print(res.summary())
print(res.params)


# In[247]:


res.plot_predict(start = '2015', end = '2025')

plt.legend(fontsize = 8)
plt.show();


# ### Working with various ML  Models for the Time Series Analysis

# #### Example 1: Vacation dataset

# In[248]:


print(df_v.head())
df_v.plot(figsize = (12,5), subplots = True)
plt.tight_layout()
plt.show()


# In[249]:


df_v.describe()


# ##### Kernel Density Plot

# In[250]:


df_v['first_diff'].plot(kind = 'kde')


# ##### Autocorrelation plot

# In[251]:


plot_acf(df_v['first_diff'])


# ##### Create lagged variables

# In[260]:


df1 = pd.DataFrame()
print(df1)


# In[261]:


# Create 12 months of lag values to predict current observations
# Shift 12 months:

for i in range(12,0,-1):
    df1['t-'+str(i)] = df_v['first_diff'].shift(i)
df1


# In[262]:


df1['t'] = df_v['first_diff'].values
df1 = df1[13:]
df1.head()


# #### Example 2: Furniture dataset

# In[263]:


df_f.head()


# In[264]:


df_f['furniture_pct_change'].plot(figsize = (12,5))


# ##### Kernal Density Plot

# In[265]:


df_f['furniture_pct_change'].plot(kind = 'density')


# ##### Autocorrelation plot

# In[ ]:


plot_acf(df_f['furniture_pct_change'])


# In[ ]:




