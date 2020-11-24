#Project 3 - Using Facebook Propher to do the Future Prediction for Crime Rate in Chicago

#Importing the Libraries
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from fbprophet import Prophet

#VISUALIZING THE DATA

# dataframes creation for both training and testing datasets 
chicago_df_1 = pd.read_csv('Chicago_Crimes_2005_to_2007.csv', error_bad_lines=False)
chicago_df_2 = pd.read_csv('Chicago_Crimes_2008_to_2011.csv', error_bad_lines=False)
chicago_df_3 = pd.read_csv('Chicago_Crimes_2012_to_2017.csv', error_bad_lines=False)

#Concatenation of dataset
chicago_df = pd.concat([chicago_df_1, chicago_df_2, chicago_df_3], ignore_index=False, axis=0)

# Let's view the head of the training dataset
chicago_df.head(10)

#Dont run belpw heatmap, it will hang the system
# Let's see how many null elements are contained in the data
#plt.figure(figsize=(10,10))
#sns.heatmap(chicago_df.isnull(), cbar = False, cmap = 'YlGnBu')

#Remove all unnecessary columns
chicago_df.drop(['Unnamed: 0', 'Case Number', 'Case Number', 'IUCR', 'X Coordinate', 'Y Coordinate','Updated On','Year', 'FBI Code', 'Beat','Ward','Community Area', 'Location', 'District', 'Latitude' , 'Longitude'], inplace=True, axis=1)

chicago_df.Date

#Rearranging the Date column in proper datetime format
chicago_df.Date = pd.to_datetime(chicago_df.Date, format = '%m/%d/%Y %I:%M:%S %p')

chicago_df

#Setting the index to be the date..... normalization step for predictions
chicago_df.index = pd.DatetimeIndex(chicago_df.Date)

chicago_df

#Visualize the diff crime types with counts
Crime_Type = chicago_df['Primary Type'].value_counts()
Crime_Type.iloc[:15] #first 15 rows
Crime_Type.iloc[:15].index

chicago_df['Primary Type'].value_counts().iloc[:15]

#Plotting the above crime types
plt.figure(figsize = (10,10))
sns.countplot(y = 'Primary Type', data = chicago_df, order = Crime_Type.iloc[:15].index)

#Plotting the Location Description
plt.figure(figsize = (10,10))
sns.countplot(y = 'Location Description', data = chicago_df, order = chicago_df['Location Description'].value_counts().iloc[:15].index)

chicago_df_10 = chicago_df.head(10)

#Printing and Plotting the yearly data (Group by year)
Yearly_data = chicago_df.Date.resample('Y').size()#for Monthly replace 'Y' with 'M' and for Quaterly with 'Q'
print(Yearly_data)

plt.plot(Yearly_data)


#PREPARING THE DATA

#resetting the index from Date to normal
chicago_prophet = chicago_df.resample('M').size().reset_index()
chicago_prophet.head(10)

#Renaming the columns
chicago_prophet.columns = ['Date', 'Crime Count']


#MAKE PREDICTIONS

#For FB Prophet we have to Rename columns one as DS and one as y as shown below 
chicago_final = chicago_prophet.rename(columns = {'Date' : 'ds' , 'Crime Count' : 'y'})

m = Prophet()
m.fit(chicago_final)

#Forecasting the future
future = m.make_future_dataframe(periods = 365)
forecast = m.predict(future)

figure = m.plot(forecast, xlabel = 'Date', ylabel = 'Crime COunt')
#plotting the monthly components
figure1 = m.plot_components(forecast)
