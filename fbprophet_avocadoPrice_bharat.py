#Project 4 - Predicting COmmodities Price(i.e. Avocado)

#Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from fbprophet import Prophet

#Importing the Dataset
dataset = pd.read_csv('avocado.csv')

#Sorting in ascendint order
dataset = dataset.sort_values('Date')

#Plotting the data with date and average price

plt.figure(figsize = (10,10))#---to make graph bit bigger
plt.plot(dataset['Date'], dataset['AveragePrice'])


plt.figure(figsize = (25,12))
sns.countplot(x = 'region', data = dataset)
plt.xticks(rotation = 45)#xticket label will be rotated by 45 degree

sns.countplot(x = 'year', data = dataset)

#We need only 2 columns for fbProphet
dataset_prophet = dataset[['Date', 'AveragePrice']]

#Prepare the Data and Predicting

dataset_final = dataset_prophet.rename(columns = {'Date' : 'ds' , 'AveragePrice' : 'y'})

m = Prophet()
m.fit(dataset_final)

#Forecasting the Future
future = m.make_future_dataframe(periods = 365)
forecast = m.predict(future)

figure = m.plot(forecast, xlabel = 'Date', ylabel = 'AveragePrice')
#plotting the monthly components
figure1 = m.plot_components(forecast)


#Part 2- Make Prediction Region Specific

dataset_part2 = dataset[dataset['region']=='West'] #Subset fo Data where Region = West
plt.plot(dataset_part2['Date'], dataset_part2['AveragePrice'])


#We need only 2 columns for fbProphet
dataset_prophet1 = dataset_part2[['Date', 'AveragePrice']]

#Prepare the Data and Predicting

dataset_final1 = dataset_prophet1.rename(columns = {'Date' : 'ds' , 'AveragePrice' : 'y'})

m = Prophet()
m.fit(dataset_final1)

#Forecasting the Future
future1 = m.make_future_dataframe(periods = 365)
forecast1 = m.predict(future1)

figure = m.plot(forecast1, xlabel = 'Date', ylabel = 'AveragePrice')
#plotting the monthly components
figure1 = m.plot_components(forecast1)

