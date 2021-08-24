#region - Part 0: Streamlit syntax, working directory

# Imports
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Streamlit app

# Tittle of the app
st.title('Categorization and Forecasting App!')

# App description
st.markdown("""
            ##
            ## Description:
            ### The aim of the code is twofold:
            #### 1- Automatically categorizing *Ahrefs* search volume data
            #### 2- Forecasting a year ahead using historic *Pytrends* data
            ##
            """)



# Asking user to provide the required data
st.markdown("## Please upload the **Ahrefs** data: ")
uploaded_file_1 = st.file_uploader("")
if uploaded_file_1 is not None:
    df_ahrefs = pd.read_excel(uploaded_file_1)

st.markdown("## Please upload the **Pytrends** data: ")
uploaded_file_2 = st.file_uploader(" ")
if uploaded_file_2 is not None:
    df_pytrends = pd.read_excel(uploaded_file_2)

# Setting working directory
parent_dir = r"C:\Users\kmane\OneDrive - Overdrive Interactive\Internship\Categorization_Forecasting"
os.chdir(parent_dir)

# Creating the output folder and asking user to provide username to be used for the output folder
st.markdown("#")
st.markdown("## Please enter your username. The final output will be a folder with the username you enter here. ")

directory = st.text_input("   ")
st.markdown("##### Note: Ignore the errors that may arise until you have finished inputing the username and datasets")
directory_1 = "./" + directory + "/"
directory_2 = directory_1 + "/Forecasts_Graphs/"
path = os.path.join(parent_dir, directory_1)
os.mkdir(path)
path = os.path.join(parent_dir, directory_2)
os.mkdir(path)

# Alert user to be patient until the code stops running
st.markdown('## Please patient until the code stops running (Top right hand side) and enjoy your data :)')

#endregion

#region - Part 1: Automated Keyword Categorization Using Supervised Machine Learning

# Importing the ahrefs data
df = df_ahrefs.copy()

# Removing non-strings from the Keyword column
df['string'] = 0
for index, item in enumerate(df['Keyword']):
    df['string'][index] = isinstance(item, str)
df = df[df['string'] == True]  ## Keeping only string keywords
df.sort_values('Keyword', inplace=True)  ## Sorting the dataframe alphabetically

# Removing duplicates
df = df[['Keyword', 'Volume', 'Position']]  ## These are the variables we are interested in keeping
df = df.groupby('Keyword', as_index=False).min(
    'Position')  ## This line of code keeps the keyword with the highest ranking if there are duplicates
df = df.groupby('Keyword', as_index=False).max(
    'Volume')  ## This line of code keeps the keyword with the highest volume if there are duplicates

# Importing the training labelled data
df_train = pd.read_excel('Data/training_data.xlsx')

# Training the model
# Imports
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier

# Training data
X_train = df_train['Keyword']
y_train = df_train['Category']

# Transforming the string data with Tfidf vectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

# Fitting the model
clf = RandomForestClassifier().fit(X_train_tfidf, y_train)
df[str(clf)] = clf.predict(count_vect.transform(df['Keyword']))

# Choosing the model for categorization
df['Category'] = df['RandomForestClassifier()'].copy()  ## Here I have chosen the Random Forest as the model 

# Storing the results of the categorization
Categorized_Results = df[['Keyword', 'Volume', 'Position', 'Category']].copy()

#endregion

#region - Part 2: Preparing Pytrends data for forecasting

# Importing the data
df = df_pytrends.copy()
pytrends_df = df.copy()

## We want the data in multiple columns based on the keywords
#df = df.pivot(index='date', columns='variable', values='value')
#df = df.rename_axis(None, axis=1).reset_index()

# Saving the names of the keywords for latter use
Keywords = df.columns[1:]

# Dealing with the date format
import datetime

# convert the 'Date' column to datetime format
df['date'] = pd.to_datetime(df['date'])

# Choosing start and end dates for our sample period to be used later
start_date = df['date'].iloc[0]
end_date = df['date'].iloc[-1]
future_date = end_date + datetime.timedelta(days=365)  ## 1 -year into the future is the goal of the prediction

# extract month and year from dates
df['Day'] = [i.day for i in df['date']]
df['Month'] = [i.month for i in df['date']]
df['Year'] = [i.year for i in df['date']]

# create a sequence of numbers
df['Series'] = np.arange(1, len(df) + 1)

# drop unnecessary columns and re-arrange
df.drop(['date'], axis=1, inplace=True)

#endregion

#region - Part 3: Forecasting future search interest using Facebook Prophet 

# Data preparation
from fbprophet import Prophet

df = pytrends_df
#df = df.pivot(index='date', columns='variable', values='value')
#df = df.rename_axis(None, axis=1).reset_index()
Results = pd.DataFrame()

# Staring a loop over all keywords
for keyword in Keywords:
    df_temp = df[['date', keyword]]
    df_temp.rename(columns={'date': 'ds', keyword: 'y'}, inplace=True)


    # Forecasting using Prophet
    m = Prophet().fit(df_temp);
    future = m.make_future_dataframe(periods=52, freq="W");  ## Predicting one year into the future
    forecast = m.predict(future);
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail();

    # Visualizing
    from fbprophet.plot import plot_forecast_component

    fig = m.plot(forecast, xlabel='Date', ylabel='Weighted Value');
    ax = fig.gca();
    ax.set_title("Forecasting Future Search Traffic for the Keyword: " + "'" + keyword + "'");
    fig.savefig(directory + "/Forecasts_Graphs/" + keyword + ".png", bbox_inches='tight');

    # Generating a results file to put everything together
    forecast.rename(columns={'yhat': keyword}, inplace=True)
    Results = pd.concat([Results, forecast[['ds', keyword]]], axis=1)

# Creating a csv file named Results_prophet under the folder Output
Results = Results.rename(columns={'ds': 'date'})
Results = Results.loc[:, ~Results.columns.duplicated()]  ## Dropping the duplicated date columns
path_temp = directory + '/Prophet_Forecasts.csv'
Results.to_csv(path_temp ,index=False)

# Storing the results of Pycaret Forecast
Prophet_Results = Results.copy()

#endregion

#region - Part 4: Merging forecasted growth to categorized data

# Using only the present year and the year forecasted
df_growth = Prophet_Results.tail(104).reset_index(drop=True) 

# Grouping the two periods (future and present)
df_growth['Future'] = 1
for index, item in enumerate(df_growth['Future']):
    if index <52:
       df_growth['Future'][index]=0 
df_growth = df_growth.groupby(by='Future').mean()       

# Defining growth as the percentage change between the two periods
df_growth = df_growth.pct_change().dropna()

# Preparing for merge with training data to get the categories
df_growth = df_growth.T.reset_index().rename(columns={'index' : 'Keyword' , 1 : 'Growth'})

# Merging with training data
df_growth = pd.merge(df_growth, df_train ,on ='Keyword', how='left')

# Equally weighted average in terms of categories
df_growth = df_growth[['Category','Growth']].groupby(by='Category').mean()  
df_growth = round(df_growth,2)

# Merging the growth data to the Categorized Results from Ahrefs
Categorized_Results = pd.merge(Categorized_Results, df_growth ,on ='Category', how='left')
Categorized_Results = Categorized_Results.sort_values(['Category', 'Volume'] , ascending = (True, False))

# Keeping the categories we have forecasts on
Categorized_Results.dropna(how='any', inplace=True)

# Returning a finalized csv output file
path_temp = directory + '/Categorized_Results.csv'
Categorized_Results.to_csv(path_temp ,index=False)

#endregion
