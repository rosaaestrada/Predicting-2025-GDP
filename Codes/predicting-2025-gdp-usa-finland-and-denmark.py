#!/usr/bin/env python
# coding: utf-8

# # World Happiness Report 2015-2024

# *Keywords:* GDP Growth Rates, Machine Learning, Logistive Regression, Random Forest, Predictive Analysis, Data Analysis

# ## Project Overview and Objectives

# #### Research Question: 
# Which country, among the USA, Finland, and Denmark, is likely to experience higher GDP growth rates in 2025, using historical data from the World Happiness Report's from 2015 to 2024 while leveraging Advanced Machine Learning Algorithms for predictive analysis?

# #### Null Hypothesis (H0):
# There is no significant difference in the predicted GDP growth rates among the USA, Finland, and Denmark in 2025, using historical data from the World Happiness Reports from 2015 to 2024.

# #### Alternative Hypothesis (H1): 
# There is a significant difference in the predicted GDP growth rates among the USA, Finland, and Denmark in 2025, using historical data from the World Happiness Reports from 2015 to 2024.

# #### Mehtodology:
# This project employs a structured methodology consisting of several key stages: data cleaning, Exploratory Data Analysis (EDA), feature engineering, and feature selection. Following those steps, predictive modeling and analysis is conducted utilizing Logistic Regression and Random Forest algorithms to ensure the target variable perfroms well. Finally, the project culminates with a comprehensive predictive modeling of the selected countries.

# ## Importing the data

# In[1]:


# import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline

from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer

from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix

from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split


# ### Load in datasets individually to add 'year' column

# ### 2015

# In[2]:


# Read the 2015 WHR data from the CSV file
df_2015 = pd.read_csv('/kaggle/input/world-happiness-report/2015.csv')

# Add a 'year' column with the value 2015 for all rows
df_2015['year'] = 2015

# Replace spaces in column names with underscores
df_2015.columns = df_2015.columns.str.replace(' ', '_')

# Now, the data from the 2015 WHR report is labeled with the year 2015
df_2015.head(5)


# In[3]:


print(df_2015.columns)


# In[4]:


#Select needed features
cols = ['Country', 'Region', 'Happiness_Rank', 'Happiness_Score',
        'Economy_(GDP_per_Capita)', 'Family', 'Health_(Life_Expectancy)', 
        'Freedom', 'Trust_(Government_Corruption)',
        'Generosity', 'Dystopia_Residual', 'year']

df_2015 = df_2015[cols]
df_2015.head(5)


# In[5]:


# rename country to merge
df_2015 = df_2015.rename(columns= {'Country' : 'country'})

df_2015.head(1)


# ### 2016

# In[6]:


# Read the 2016 WHR data from the CSV file
df_2016 = pd.read_csv('/kaggle/input/world-happiness-report/2016.csv')

# Add a 'year' column with the value 2016 for all rows
df_2016['year'] = 2016

# Replace spaces in column names with underscores
df_2016.columns = df_2016.columns.str.replace(' ', '_')

# Now, the data from the 2016 WHR report is labeled with the year 2016
df_2016.head()


# In[7]:


print(df_2016.columns)


# In[8]:


#Select needed features
cols = ['Country', 'Region', 'Happiness_Rank', 'Happiness_Score',
       'Economy_(GDP_per_Capita)', 'Family', 'Health_(Life_Expectancy)',
       'Freedom', 'Trust_(Government_Corruption)', 'Generosity',
       'Dystopia_Residual', 'year']

df_2016 = df_2016[cols]
df_2016.head(5)


# In[9]:


# rename country to merge
df_2016 = df_2016.rename(columns= {'Country' : 'country'})

df_2016.head(1)


# ### 2017

# In[10]:


# Read the 2017 WHR data from the CSV file
df_2017 = pd.read_csv('/kaggle/input/world-happiness-report/2017.csv')

# Add a 'year' column with the value 2017 for all rows
df_2017['year'] = 2017

# Replace spaces in column names with underscores
df_2017.columns = df_2017.columns.str.replace(' ', '_')

# Now, the data from the 2017 WHR report is labeled with the year 2017
df_2017.head()


# In[11]:


print(df_2017.columns)


# In[12]:


#Select needed features
cols = ['Country', 'Happiness.Rank', 'Happiness.Score',
        'Economy..GDP.per.Capita.', 'Family',
        'Health..Life.Expectancy.', 'Freedom', 'Generosity',
        'Trust..Government.Corruption.', 'Dystopia.Residual', 'year']

df_2017 = df_2017[cols]
df_2017.head(5)


# In[13]:


# rename country to merge
df_2017 = df_2017.rename(columns= {'Country' : 'country'})

df_2017.head(1)


# ### 2018

# In[14]:


# Read the 2018 WHR data from the CSV file
df_2018 = pd.read_csv('/kaggle/input/world-happiness-report/2018.csv')

# Add a 'year' column with the value 2018 for all rows
df_2018['year'] = 2018

# Replace spaces in column names with underscores
df_2018.columns = df_2018.columns.str.replace(' ', '_')

# Now, the data from the 2018 WHR report is labeled with the year 2018
df_2018.head()


# In[15]:


print(df_2018.columns)


# In[16]:


#Select needed features
cols = ['Overall_rank', 'Country_or_region', 'Score', 'GDP_per_capita',
       'Social_support', 'Healthy_life_expectancy',
       'Freedom_to_make_life_choices', 'Generosity',
       'Perceptions_of_corruption', 'year']

df_2018 = df_2018[cols]
df_2018.head(5)


# In[17]:


# rename country to merge
df_2018 = df_2018.rename(columns= {'Country_or_region' : 'country'})

df_2018.head(1)


# ### 2019

# In[18]:


# Read the 2019 WHR data from the CSV file
df_2019 = pd.read_csv('/kaggle/input/world-happiness-report/2019.csv')

# Add a 'year' column with the value 2019 for all rows
df_2019['year'] = 2019

# Replace spaces in column names with underscores
df_2019.columns = df_2019.columns.str.replace(' ', '_')

# Now, the data from the 2019 WHR report is labeled with the year 2019
df_2019.head()


# In[19]:


print(df_2019.columns)


# In[20]:


#Select needed features
cols = ['Overall_rank', 'Country_or_region', 'Score', 'GDP_per_capita',
       'Social_support', 'Healthy_life_expectancy',
       'Freedom_to_make_life_choices', 'Generosity',
       'Perceptions_of_corruption', 'year']

df_2019 = df_2019[cols]
df_2019.head(5)


# In[21]:


# rename country to merge
df_2019 = df_2019.rename(columns= {'Country_or_region' : 'country'})

df_2019.head(1)


# ### 2020

# In[22]:


# Read the 2020 WHR data from the CSV file
df_2020 = pd.read_csv('/kaggle/input/world-happiness-report/2020.csv')

# Add a 'year' column with the value 2020 for all rows
df_2020['year'] = 2020

# Replace spaces in column names with underscores
df_2020.columns = df_2020.columns.str.replace(' ', '_')

# Now, the data from the 2020 WHR report is labeled with the year 2020
df_2020.head()


# In[23]:


print(df_2020.columns)


# In[24]:


#Select needed features
cols = ['Country_name', 'Regional_indicator', 'Ladder_score',
       'Logged_GDP_per_capita', 'Social_support', 'Healthy_life_expectancy',
       'Freedom_to_make_life_choices', 'Generosity',
       'Perceptions_of_corruption', 'Ladder_score_in_Dystopia',
       'Dystopia_+_residual', 'year']

df_2020 = df_2020[cols]
df_2020.head(5)


# In[25]:


# rename country to merge
df_2020 = df_2020.rename(columns= {'Country_name' : 'country'})

df_2020.head(1)


# ### 2021

# In[26]:


# Read the 2021 WHR data from the CSV file
df_2021 = pd.read_csv('/kaggle/input/world-happiness-report/2021.csv')

# Add a 'year' column with the value 2021 for all rows
df_2021['year'] = 2021

# Replace spaces in column names with underscores
df_2021.columns = df_2021.columns.str.replace(' ', '_')

# Now, the data from the 2021 WHR report is labeled with the year 2021
df_2021.head()


# In[27]:


print(df_2021.columns)


# In[28]:


#Select needed features
cols = ['Country_name', 'Regional_indicator', 'Ladder_score',
       'Logged_GDP_per_capita', 'Social_support', 'Healthy_life_expectancy',
       'Freedom_to_make_life_choices', 'Generosity',
       'Perceptions_of_corruption', 'Ladder_score_in_Dystopia',
       'Dystopia_+_residual', 'year']

df_2021 = df_2021[cols]
df_2021.head(5)


# In[29]:


# rename country to merge
df_2021 = df_2021.rename(columns= {'Country_name' : 'country'})

df_2021.head(1)


# ### 2022

# In[30]:


# Read the 2022 WHR data from the CSV file
df_2022 = pd.read_csv('/kaggle/input/world-happiness-report/2022.csv')

# Add a 'year' column with the value 2022 for all rows
df_2022['year'] = 2022

# Replace spaces in column names with underscores
df_2022.columns = df_2022.columns.str.replace(' ', '_')

# Now, the data from the 2022 WHR report is labeled with the year 2022
df_2022.head()


# In[31]:


print(df_2022.columns)


# In[32]:


#Select needed features
cols = ['RANK', 'Country', 'Happiness_score', 'year']

df_2022 = df_2022[cols]
df_2022.head(5)


# In[33]:


# rename country to merge
df_2022 = df_2022.rename(columns= {'Country' : 'country'})

df_2022.head(1)


# ### 2023

# In[34]:


# Read the 2023 WHR data from the CSV file
df_2023 = pd.read_csv('/kaggle/input/world-happiness-report-2023/WHR2023.csv')

# Add a 'year' column with the value 2023 for all rows
df_2023['year'] = 2023

# Replace spaces in column names with underscores
df_2023.columns = df_2023.columns.str.replace(' ', '_')

# Now, the data from the 2023 WHR report is labeled with the year 2023
df_2023.head()


# In[35]:


print(df_2023.columns)


# In[36]:


#Select needed features
cols = ['Country_name', 'Ladder_score', 'Logged_GDP_per_capita', 'Social_support', 
        'Healthy_life_expectancy', 'Freedom_to_make_life_choices', 'Generosity',
        'Perceptions_of_corruption', 'Ladder_score_in_Dystopia',
        'Dystopia_+_residual', 'year']

df_2023 = df_2023[cols]
df_2023.head(5)


# In[37]:


# rename country to merge
df_2023 = df_2023.rename(columns= {'Country_name' : 'country'})

df_2023.head(1)


# ### 2024

# In[38]:


# Read the 2024 WHR data from the CSV file
df_2024 = pd.read_csv('/kaggle/input/world-happiness-report-2024/WHR2024.csv')

# Add a 'year' column with the value 2024 for all rows
df_2024['year'] = 2024

# Replace spaces in column names with underscores
df_2024.columns = df_2024.columns.str.replace(' ', '_')

# Now, the data from the 20202423 WHR report is labeled with the year 2024
df_2024.head()


# In[39]:


print(df_2024.columns)


# In[40]:


#Select needed features
cols = ['Country_name', 'Ladder_score', 'Dystopia_+_residual', 'year']

df_2024 = df_2024[cols]
df_2024.head(5)


# In[41]:


# rename country to merge
df_2024 = df_2024.rename(columns= {'Country_name' : 'country'})

df_2024.head(1)


# ## Combine datasets into one

# In[42]:


# List of DataFrames to combine
dataframes = [df_2015, df_2016, df_2017, df_2018, df_2019, df_2020, df_2021, df_2022, df_2023, df_2024]

# Combine the DataFrames into one DataFrame named 'WHR'
WHR = pd.concat(dataframes, ignore_index=True)

WHR.head(5)


# In[43]:


# Print the updated column names to verify the change
print(WHR.columns)


# ## Combine similar columns into new features

# In[44]:


# Create a new column 'region' by combining 'Region', and 'Regional_indicator'

# Use fillna() to handle missing values and keep only the non-null values
WHR['region'] = WHR['Region'].fillna(WHR['Regional_indicator'])

# Display the first 5 rows of the DataFrame with the new 'region' column
print(WHR[['region']].head(5))


# In[45]:


# Create a new column 'happiness_rank' by combining 'Happiness_Rank', 'Happiness.Rank', 'Overall_rank', and 'RANK'

# Use fillna() to handle missing values and keep only the non-null values
WHR['happiness_rank'] = WHR['Happiness_Rank'].fillna(WHR['Happiness.Rank']).fillna(WHR['Overall_rank']).fillna(WHR['RANK'])

# Display the first 5 rows of the DataFrame with the new 'happiness_rank' column
print(WHR[['happiness_rank']].head(5))


# In[46]:


# Create a new column 'new_happiness_score' by combining 'Happiness_Score', 'Happiness.Score', 'Score', 'Ladder_score', and 'Happiness_score'

# Use fillna() to handle missing values and keep only the non-null values
WHR['new_happiness_score'] = WHR['Happiness_Score'].fillna(WHR['Happiness.Score']).fillna(WHR['Score']).fillna(WHR['Ladder_score']).fillna(WHR['Happiness_score'])

# Display the first 5 rows of the DataFrame with the new 'new_happiness_score' column
print(WHR[['new_happiness_score']].head(5))


# In[47]:


# Create a new column 'economy' by combining 'Economy_(GDP_per_Capita)', 'Economy..GDP.per.Capita.', 'GDP_per_capita', and 'Logged_GDP_per_capita'

# Use fillna() to handle missing values and keep only the non-null values
WHR['economy'] = WHR['Economy_(GDP_per_Capita)'].fillna(WHR['Economy..GDP.per.Capita.']).fillna(WHR['GDP_per_capita']).fillna(WHR['Logged_GDP_per_capita'])

# Display the first 5 rows of the DataFrame with the new 'economy' column
print(WHR[['economy']].head(5))


# In[48]:


# Create a new column 'social_support' by combining 'Family' and 'Social_support'

# Use fillna() to handle missing values and keep only the non-null values
WHR['social_support'] = WHR['Family'].fillna(WHR['Social_support'])

# Display the first 5 rows of the DataFrame with the new 'social_support' column
print(WHR[['social_support']].head(5))


# In[49]:


# Create a new column 'life_expectancy' by combining 'Health_(Life_Expectancy)', 'Health..Life.Expectancy.', and 'Healthy_life_expectancy'

# Use fillna() to handle missing values and keep only the non-null values
WHR['life_expectancy'] = WHR['Health_(Life_Expectancy)'].fillna(WHR['Health..Life.Expectancy.']).fillna(WHR['Healthy_life_expectancy'])

# Display the first 5 rows of the DataFrame with the new 'life_expectancy' column
print(WHR[['life_expectancy']].head(5))


# In[50]:


# Create a new column 'freedom' by combining 'Health_(Life_Expectancy)' and 'Freedom_to_make_life_choices'

# Use fillna() to handle missing values and keep only the non-null values
WHR['freedom'] = WHR['Health_(Life_Expectancy)'].fillna(WHR['Freedom_to_make_life_choices'])

# Display the first 5 rows of the DataFrame with the new 'freedom' column
print(WHR[['freedom']].head(5))


# In[51]:


# Create a new column 'corruption' by combining 'Trust_(Government_Corruption)', 'Trust..Government.Corruption.', and 'Perceptions_of_corruption'

# Use fillna() to handle missing values and keep only the non-null values
WHR['corruption'] = WHR['Trust_(Government_Corruption)'].fillna(WHR['Trust..Government.Corruption.']).fillna(WHR['Perceptions_of_corruption'])

# Display the first 5 rows of the DataFrame with the new 'corruption' column
print(WHR[['corruption']].head(5))


# In[52]:


# Create a new column 'dystopia' by combining 'Dystopia_Residual', 'Dystopia.Residual', 'Ladder_score_in_Dystopia', and 'Dystopia_+_residual'

# Use fillna() to handle missing values and keep only the non-null values
WHR['dystopia'] = WHR['Dystopia_Residual'].fillna(WHR['Dystopia.Residual']).fillna(WHR['Ladder_score_in_Dystopia']).fillna(WHR['Dystopia_+_residual'])

# Display the first 5 rows of the DataFrame with the new 'dystopia' column
print(WHR[['dystopia']].head(5))


# In[53]:


# rename generosity to keep it similar to the rest
WHR = WHR.rename(columns= {'Generosity' : 'generosity'})


# In[54]:


# Drop repetative columns and keep only the new features with combines columns
columns_to_drop = [
    'Region', 'Regional_indicator', 'Happiness_Rank', 'Happiness.Rank', 
    'Overall_rank', 'RANK', 'Happiness_Score', 'Happiness.Score', 'Score', 
    'Ladder_score', 'Happiness_score', 'Economy_(GDP_per_Capita)', 
    'Economy..GDP.per.Capita.', 'GDP_per_capita', 'Logged_GDP_per_capita', 'Family', 
    'Social_support', 'Health_(Life_Expectancy)', 'Health..Life.Expectancy.', 
    'Healthy_life_expectancy', 'Freedom', 'Freedom_to_make_life_choices',
    'Trust_(Government_Corruption)', 'Trust..Government.Corruption.', 
    'Perceptions_of_corruption', 'Dystopia_Residual', 'Dystopia.Residual', 
    'Ladder_score_in_Dystopia', 'Dystopia_+_residual'
]

# Drop the columns
WHR.drop(columns=columns_to_drop, axis=1, inplace=True)

# Verify the change
print(WHR.columns)


# In[55]:


WHR = WHR.rename(columns= {'new_happiness_score' : 'happiness_score'})


# In[56]:


print(WHR.dtypes)


# In[57]:


# Convert 'happiness_score' column to numeric
WHR['happiness_score'] = pd.to_numeric(WHR['happiness_score'], errors='coerce')

# Check the data types after conversion
print(WHR.dtypes)


# In[58]:


WHR.head()


# In[59]:


features_of_interest = ['happiness_score', 'economy', 'social_support', 'life_expectancy', 
                        'freedom', 'generosity', 'corruption']

plt.figure(figsize=(10, 6))
sns.barplot(data=WHR[features_of_interest], palette='viridis')
plt.xlabel('Features')
plt.ylabel('Values')
plt.title('Bar Plot of Selected Features in WHR Dataset')
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.show()


# In[60]:


WHR.shape


# ### Variables used in  the project:
# **country**
# 
# **region**: 
# *The geographical area in which a country is located*
# 
# **happiness_rank**: 
# *Ranking of countries based on their happiness scores with 1 being the happiest country
# 
# **happiness_score**:
# *The numeric representation of a countr's overall happienss level, based on the following variables*
# 
# **economy (TARGET VARIABLE)**: 
# *(GDP Per Capita), How much each country produces divided by the number of people in the country?*
# 
# **social_support**: 
# *If you were in trouble, do you have relatives or friends you can count on to help you whenever you need them, or not?*
# 
# **life_expectancy**: 
# *(Healthy Life Expectancy), How is your physical and mental health?* 
# 
# **freeedom**: 
# *(Freedom to make life choices), Are you satisfied or dissatisfied with your freedom to choose what you do with your life?*
# 
# **generosity**: *Have you donated money to charity in the past month?*
# 
# **corruption**: 
# *(Perception of corruption), looks at both governments and businesses*
# 
# **dystopia**: 
# *A benchmark/imaginary country that ahs the world's least-happy people. no country performs more poorly than Dystopia*
# 
# 
# 
# - Infomration on these variables received from the WHR 2023 pdf report

# In[61]:


# List of countries to keep
countries_to_keep = ['United States', 'Finland', 'Denmark']

# List of regions to keep
regions_to_keep = ['North America', 'Western Europe']

# Filter the DataFrame to include only the specified countries and regions
WHR2 = WHR[WHR['country'].isin(countries_to_keep) & WHR['region'].isin(regions_to_keep)]

# Check the filtered DataFrame
WHR2.head()


# ## Numeric Variable Analysis

# In[62]:


#list numeric features along with their statistical description
des = WHR.select_dtypes(exclude=['object']).describe().round(decimals=2).transpose()
print(des.to_string())


# In[63]:


#Plotting histograms
WHR[['generosity']].hist(bins=50, figsize=(13, 10), layout=(2,2), color='pink', edgecolor='black')


# In[64]:


#Plotting histograms
WHR[['year']].hist(bins=50, figsize=(13, 10), layout=(2,2), color='pink', edgecolor='black')


# In[65]:


#Plotting histograms
WHR[['happiness_rank']].hist(bins=50, figsize=(13, 10), layout=(2,2), color='pink', edgecolor='black')


# In[66]:


#Plotting histograms
WHR[['happiness_score']].hist(bins=50, figsize=(13, 10), layout=(2,2), color='pink', edgecolor='black')


# In[67]:


#Plotting histograms
WHR[['economy']].hist(bins=50, figsize=(13, 10), layout=(2,2), color='pink', edgecolor='black')


# In[68]:


#Plotting histograms
WHR[['social_support']].hist(bins=50, figsize=(13, 10), layout=(2,2), color='pink', edgecolor='black')


# In[69]:


#Plotting histograms
WHR[['life_expectancy']].hist(bins=50, figsize=(13, 10), layout=(2,2), color='pink', edgecolor='black')


# In[70]:


#Plotting histograms
WHR[['freedom']].hist(bins=50, figsize=(13, 10), layout=(2,2), color='pink', edgecolor='black')


# In[71]:


#Plotting histograms
WHR[['corruption']].hist(bins=50, figsize=(13, 10), layout=(2,2), color='pink', edgecolor='black')


# In[72]:


#Plotting histograms
WHR[['dystopia']].hist(bins=50, figsize=(13, 10), layout=(2,2), color='pink', edgecolor='black')


# *NOTE: Because the variables are scores, removing outliers may hinder results*

# ## Categorical Variable analysis

# In[73]:


# Unique categories for each categorical variable
WHR.select_dtypes('object').apply(pd.Series.nunique, axis = 0)


# In[74]:


# Count the unique categories in the 'country' column
country_counts = WHR['country'].value_counts()

# Plotting the bar plot
plt.figure(figsize=(28, 20))
country_counts.plot(kind='bar')

plt.title('Number of Entries for Each Country')
plt.xlabel('Country')
plt.ylabel('Count')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.show()


# In[75]:


# Count the unique categories in the 'region' column
region_counts = WHR['region'].value_counts()

# Plotting the bar plot
plt.figure(figsize=(15, 10))
region_counts.plot(kind='bar', cmap= 'cividis')

plt.title('Number of Entries for Each Region')
plt.xlabel('Region')
plt.ylabel('Count')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.show()


# ## Handle Missing Values

# In[76]:


print(WHR.dtypes)


# In[77]:


# Check for missing values
missing_values = WHR.isnull().sum()

# Print the count of missing values in each column
print(missing_values)


# In[78]:


print(WHR.shape)


# ### KNN imputation for numeric missing values
# (takes the nearest neighbors)

# In[79]:


# List of numeric variables to impute
numeric_vars = [
    'generosity', 
    'happiness_rank', 
    'happiness_score', 
    'economy', 
    'social_support', 
    'life_expectancy', 
    'freedom', 
    'corruption', 
    'dystopia'
]

# Initialize the KNN imputer
imputer = KNNImputer(n_neighbors=20)

# Impute missing values
WHR[numeric_vars] = imputer.fit_transform(WHR[numeric_vars])

# Verify the changes
print(WHR[numeric_vars].isnull().sum())


# In[80]:


# Calculate the mode of the 'region' variable
mode_region = WHR['region'].mode()[0]

# Impute missing values in the 'region' column with the mode
WHR['region'] = WHR['region'].fillna(mode_region)

# Verify the changes
print(WHR['region'].isnull().sum())


# In[81]:


WHR2 = WHR.copy()

WHR2.head()


# ## Check for duplicated rows

# In[82]:


# Check for duplicate rows
duplicate_rows = WHR2.duplicated()

# Calculate the number of duplicate rows
num_duplicates = duplicate_rows.sum()
print(f"Number of duplicate rows in the WHR4 dataset: {num_duplicates}")

# Display the duplicate rows
if num_duplicates > 0:
    print("Duplicate rows in the WHR4 dataset:")
    duplicates = WHR2[duplicate_rows]
    print(duplicates)


# ## Feature Engineering

# ### economy

# In[83]:


WHR2['economy'].unique()


# In[84]:


# Create a integer part of 'economy'
WHR2['economy'] = WHR2['economy'].astype(int)

# Print unique values to verify the grouping
print(WHR2['economy'].unique())


# In[85]:


value_counts = WHR2['economy'].value_counts()

# Sort the value counts in ascending order
sorted_value_counts = value_counts.sort_index(ascending=True)

# Plot the distribution using seaborn
plt.figure(figsize=(12, 8))  # Adjust the figure size as needed
sns.barplot(x=sorted_value_counts.index, y=value_counts.values, palette='viridis')  # Change color palette if needed

plt.xlabel('Economy')
plt.ylabel('Count')

plt.tight_layout()  # Adjust layout
plt.show()


# In[86]:


value_counts = WHR2['economy'].value_counts()

# Plot the distribution using seaborn without sorting
plt.figure(figsize=(12, 8))
sns.barplot(x=value_counts.index, y=value_counts.values, palette='viridis')  # Change color palette if needed

plt.xlabel('Economy')
plt.ylabel('Count')

plt.tight_layout()  # Adjust layout
plt.show()


# ## Exploratory Data Analysis

# In[87]:


WHR2.head(2)


# ### year

# In[88]:


sns.set(style= "whitegrid")

# Create bar plot
plt.figure(figsize= (10, 7))
ax= sns.countplot(x= 'year', hue= 'economy', data= WHR2)

# Move the legend to upper right
ax.legend(loc='right', bbox_to_anchor= (1.1,0.5))

# Set the title and labels
plt.title('Distribution of Economy by Year')
plt.xlabel('Year')
plt.ylabel('Count')

plt.show()


# ### region

# In[89]:


sns.set(style= "whitegrid")

# Create bar plot
plt.figure(figsize= (8, 4))
ax= sns.countplot(x= 'region', hue= 'economy', data= WHR2)

# Move the legend to upper right
ax.legend(loc='upper right')

# Set the title and labels
plt.title('Distribution of Economy by Region')
plt.xlabel('Region')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')  # Rotate x-labels for better readability and align them to the right

plt.show()


# ## One-hot encoding

# In[90]:


print(WHR2.dtypes)


# * country.............................object <- Categorical
# * generosity......................float64 <- Numeric
# * year......................................int64 <- Numeric
# * region...............................object <- Categorical
# * happiness_rank...........float64 <- Numeric
# * happiness_score..........float64 <- Numeric
# * economy............................int32 <- Numeric
# * social_support..............float64 <- Numeric 
# * life_expectancy.............float64 <- Numeric
# * freedom..........................float64 <- Numeric
# * corruption.......................float64 <- Numeric 
# * dystopia..........................float64 <- Numeric

# In[91]:


# Perform one-hot encoding for 'country' and 'region' columns
WHR3 = pd.get_dummies(WHR2, columns=['country', 'region'])


# In[92]:


# Check the shape of the DataFrame after one-hot encoding
print(WHR3.shape)


# ## Feature Selection 
# 
# (Not needed with this research question but great to see the relationship among variables)

# In[93]:


# Correlation Matrix Heatmap
corr = WHR3. corr()

plt.figure(figsize= (15,15))
heatmap = sns.heatmap(corr, vmin= -1, vmax= 1, annot = False, cmap= 'ocean', 
                      cbar= True)

plt.show()


# In[94]:


# Set the correlation threshold
threshold = 0.2

# Find features that are correlated 
correlated_pairs = []
for i in range(len(corr.columns)):
    for j in range(i+1, len(corr.columns)):
        col1 = corr.columns[i]
        col2 = corr.columns[j]
        corr_value = corr.iloc[i, j]
        if abs(corr_value) >= threshold:  # Check if correlation meets threshold
            correlated_pairs.append((col1, col2, corr_value))
        
# Print out correlated pairs with their correlation coefficient
print("Correlated pairs with their correlation coefficient (above threshold):")
for col1, col2, corr_value in correlated_pairs:
    print(f"{col1} and {col2} have a correlation of {corr_value:.2f}")


# In[95]:


#Set up threshold
target_variable = 'economy'
threshold = 0.0

#Find features with significant correlation with the target variable
significant_correlations = corr[target_variable].drop(target_variable).where(lambda x: abs(x) > threshold).dropna()

#Print
for feature, corr_value in significant_correlations.items():
    print(f"{feature} and {target_variable} have a correlation of {corr_value:.2f}")


# In[96]:


# Calculate correlations
correlation_matrix = WHR3.corr()

# Filter correlations with 'loan_status'
correlation_with_loan_status = correlation_matrix['economy'].drop('economy')

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_with_loan_status.to_frame(), cmap='BrBG', annot=False, fmt=".2f", vmin=-1, vmax=1, cbar=True)
plt.title("Correlation with Economy")
plt.xlabel("Features")
plt.ylabel("Economy")
plt.show()


# In[97]:


# Select features based on correlation outcome
selected_features = ['year', 'happiness_rank', 'life_expectancy', 'freedom', 'corruption',
                     'country_Australia', 'country_Canada', 'country_United States',
                     'country_Hong Kong S.A.R. of China', 'country_Maldives',
                     'region_Middle East and North Africa', 'region_Sub-Saharan Africa',
                     'economy'] 

# Filter the dataset to keep only the selected features
WHR4 = WHR3[selected_features]

# Display the first few rows of the filtered dataset
WHR4.head()


# ## Machine Learning Algorithms

# ### Linear Regression & Random Forest models

# In[98]:


# Define features (independent variables) and target variable (dependent variable)
X = WHR4.drop(columns=['economy'])  # Features
Y = WHR4['economy']  # Target variable

# Split the data into training and testing sets (75-25 split)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)


# In[99]:


# Create a pipeline that standardizes the data then applies linear regression
pipeline = Pipeline(steps=[('scaler', StandardScaler()), ('regressor', LinearRegression())])

# Fit the pipeline to the training data
pipeline.fit(X_train, Y_train)


# In[100]:


# Random forest regression model
random_forest_model = RandomForestRegressor(n_estimators=100, random_state=1234)

# Fit the model to the training data
random_forest_model.fit(X_train, Y_train)


# In[101]:


# Make predictions on the testing data
Y_pred_logistic = pipeline.predict(X_test)
Y_pred_rf = random_forest_model.predict(X_test)


# In[102]:


# Compute evaluation metrics for logistic regression model
mae_logistic = mean_absolute_error(Y_test,Y_pred_logistic)
mse_logistic = mean_squared_error(Y_test, Y_pred_logistic)
r2_logistic = r2_score(Y_test, Y_pred_logistic)

# Compute evaluation metrics for random forest model
mae_rf = mean_absolute_error(Y_test, Y_pred_rf)
mse_rf = mean_squared_error(Y_test, Y_pred_rf)
r2_rf = r2_score(Y_test, Y_pred_rf)

# Print evaluation metrics
print("Evaluation metrics for Logistic Regression:")
print("Mean Absolute Error:", mae_logistic)
print("Mean Squared Error:", mse_logistic)
print("R-squared (R^2):", r2_logistic)
print()

print("Evaluation metrics for Random Forest:")
print("Mean Absolute Error:", mae_rf)
print("Mean Squared Error:", mse_rf)
print("R-squared (R^2):", r2_rf)


# **How close the model's predictions are to the actual happiness score**
# * <u>Mean Absolute Error (MAE):</u> Lower values indicate better performance
#     * Logistic Regression: 0.364247597472116
#     * Random Forest: 0.2863227513227513
# 
# * <u>Mean Squared Error (MSE):</u> Lower values indicate better performance
#     * Logistic Regression: 0.26206355385445756
#     * Random Forest: 0.2083251322751323
# 
# **Indicates the proportion of variance in the happiness scores that is explained by the model**
# * <u>R-Squared (R^2):</u> A higher R-squared value indicates better model fit to the data 
#     * Logistic Regression: 0.983245594541835
#     * Random Forest: 0.9866812317778386

# In[103]:


#Get feature importances
importances = random_forest_model.feature_importances_

#Create a DataFrame to view the features and their importance scores
features_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})

#Sort the DataFrame to see the most important features at the top
features_df.sort_values(by='Importance', ascending=False, inplace=True)

print(features_df)


# ## Evaluation on the target variable perfromance

# ### Random Forest model perfromed better in predicting economy (GDP growth rates)
# **Logistic Regression**
# * Mean Absolute Error: 0.364
# * Mean squared Error: 0.262
# * R-squared (R^2): 0.983
# 
# **Random Forest**
# * Mean Absolute Error: 0.286
# * Mean Squared Error: 0.208
# * R-squared (R^2): 0.987

# * R-squared values for both models are close to 1, indicating that the models explain a high percentage of the variance in the target variable (economic- GDP growth rate)
# 
# * Demonstrate strong potential in predicting economic indicators like GDP growth rates 

# ## Find which countries are likely to have higher GDP growth rates (economy) in 2025

# In[104]:


WHR5 = WHR2.copy()

WHR5.head()


# In[105]:


print(WHR5.dtypes)


# ## Average the score and find the top two countries

# #### These two variables are the ones I will use in the research question

# In[106]:


# Calculate the average happiness score for each country
average_happiness_score = WHR5.groupby('country')['happiness_score'].mean()

# Sort the countries based on the average happiness score
sorted_average_scores = average_happiness_score.sort_values(ascending=False)

# Select the top two countries
top_two_countries = sorted_average_scores.head(2)

# Display the top two countries and their average happiness scores
print("Top Two Countries with Highest Average Happiness Score:")
print(top_two_countries)


# ## Extract countries into different datasets - Might use (Havent yet)

# In[107]:


# Group the dataset by country
grouped_country = WHR5.groupby('country')

# Extract data for the United States, Finland, and Denmark
usa = grouped_country.get_group('United States')
finland = grouped_country.get_group('Finland')
denmark = grouped_country.get_group('Denmark')

# Now you have separate datasets for each country
print("United States data:")
print(usa.head())

print("\nFinland data:")
print(finland.head())

print("\nDenmark data:")
print(denmark.head())


# ## One-hot encoding for 3 countries

# In[108]:


# Perform one-hot encoding for 'country' and 'region' columns
usa2 = pd.get_dummies(usa, columns=['country', 'region'])


# In[109]:


# Perform one-hot encoding for 'country' and 'region' columns
finland2 = pd.get_dummies(finland, columns=['country', 'region'])


# In[110]:


# Perform one-hot encoding for 'country' and 'region' columns
denmark2 = pd.get_dummies(denmark, columns=['country', 'region'])


# ### usa2

# In[111]:


# Correlation Matrix Heatmap
corr_usa = usa2. corr()

plt.figure(figsize= (15,15))
heatmap = sns.heatmap(corr_usa, vmin= -1, vmax= 1, annot = False, cmap= 'ocean', cbar= True)

plt.show()


# ### finland2

# In[112]:


# Correlation Matrix Heatmap
corr_finland = finland2. corr()

plt.figure(figsize= (15,15))
heatmap = sns.heatmap(corr_finland, vmin= -1, vmax= 1, annot = False, cmap= 'ocean', cbar= True)

plt.show()


# ### denmark2

# In[113]:


# Correlation Matrix Heatmap
corr_denmark = denmark2. corr()

plt.figure(figsize= (15,15))
heatmap = sns.heatmap(corr_denmark, vmin= -1, vmax= 1, annot = False, cmap= 'ocean', cbar= True)

plt.show()


# ### usa2

# In[114]:


# Calculate correlations
corr_matrix_usa = usa2.corr()

# Filter correlations with 'economy'
correlation_with_economy_usa = corr_matrix_usa['economy'].drop('economy')

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_with_economy_usa.to_frame(), cmap='BrBG', annot=False, fmt=".2f", vmin=-1, vmax=1, cbar=True)
plt.title("Correlation with Economy")
plt.xlabel("Features")
plt.ylabel("Economy")
plt.show()


# ### finland2

# In[115]:


# Calculate correlations
corr_matrix_finland = finland2.corr()

# Filter correlations with 'economy'
correlation_with_economy_fin = corr_matrix_finland['economy'].drop('economy')

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_with_economy_fin.to_frame(), cmap='BrBG', annot=False, fmt=".2f", vmin=-1, vmax=1, cbar=True)
plt.title("Correlation with Economy")
plt.xlabel("Features")
plt.ylabel("Economy")
plt.show()


# ### denmark2

# In[116]:


# Calculate correlations
corr_matrix_denmark = denmark2.corr()

# Filter correlations with 'loan_status'
correlation_with_economy_den = corr_matrix_denmark['economy'].drop('economy')

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_with_economy_den.to_frame(), cmap='BrBG', annot=False, fmt=".2f", vmin=-1, vmax=1, cbar=True)
plt.title("Correlation with Economy")
plt.xlabel("Features")
plt.ylabel("Economy")
plt.show()


# ## Machine Learning Algorithms
# ## Logistic Regression & Random Forest

# ### usa2 - small 'x' & small 'y'

# In[117]:


# Define features (independent variables) and target variable (dependent variable)
x  = usa2.drop(columns=['economy'])  # Features
y = usa2['economy']  # Target variable

# Split the data into training and testing sets (75-25 split)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)


# In[118]:


# Create a pipeline that standardizes the data then applies linear regression
pipeline = Pipeline(steps=[('scaler', StandardScaler()), ('regressor', LinearRegression())])

# Fit the pipeline to the training data
pipeline.fit(x_train, y_train)


# In[119]:


# Random forest regression model
random_forest_model = RandomForestRegressor(n_estimators=100, random_state=1234)

# Fit the model to the training data
random_forest_model.fit(x_train, y_train)


# In[120]:


# Make predictions on the testing data
y_pred_logistic = pipeline.predict(x_test)
y_pred_rf       = random_forest_model.predict(x_test)


# In[121]:


# Compute evaluation metrics for logistic regression model
mae_logistic = mean_absolute_error(y_test, y_pred_logistic)
mse_logistic = mean_squared_error(y_test, y_pred_logistic)
r2_logistic = r2_score(y_test, y_pred_logistic)

# Compute evaluation metrics for random forest model
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# Print evaluation metrics
print("Evaluation metrics for Logistic Regression:")
print("Mean Absolute Error:", mae_logistic)
print("Mean Squared Error:", mse_logistic)
print("R-squared (R^2):", r2_logistic)
print()

print("Evaluation metrics for Random Forest:")
print("Mean Absolute Error:", mae_rf)
print("Mean Squared Error:", mse_rf)
print("R-squared (R^2):", r2_rf)


# ### finland2 - big 'XX' & big 'YY'

# In[122]:


# Define features (independent variables) and target variable (dependent variable)
XX = finland2.drop(columns=['economy'])  # Features
YY = finland2['economy']  # Target variable

# Split the data into training and testing sets (75-25 split)
XX_train, XX_test, YY_train, YY_test = train_test_split(XX, YY, test_size=0.25, random_state=42)


# In[123]:


# Create a pipeline that standardizes the data then applies linear regression
pipeline = Pipeline(steps=[('scaler', StandardScaler()), ('regressor', LinearRegression())])

# Fit the pipeline to the training data
pipeline.fit(XX_train, YY_train)


# In[124]:


# Random forest regression model
random_forest_model = RandomForestRegressor(n_estimators=100, random_state=1234)

# Fit the model to the training data
random_forest_model.fit(XX_train, YY_train)


# In[125]:


# Make predictions on the testing data
YY_pred_logistic = pipeline.predict(XX_test)
YY_pred_rf       = random_forest_model.predict(XX_test)


# In[126]:


# Compute evaluation metrics for logistic regression model
mae_logistic = mean_absolute_error(YY_test, YY_pred_logistic)
mse_logistic = mean_squared_error(YY_test, YY_pred_logistic)
r2_logistic  = r2_score(YY_test, YY_pred_logistic)

# Compute evaluation metrics for random forest model
mae_rf = mean_absolute_error(YY_test, YY_pred_rf)
mse_rf = mean_squared_error(YY_test, YY_pred_rf)
r2_rf  = r2_score(YY_test, YY_pred_rf)

# Print evaluation metrics
print("Evaluation metrics for Logistic Regression:")
print("Mean Absolute Error:", mae_logistic)
print("Mean Squared Error:", mse_logistic)
print("R-squared (R^2):", r2_logistic)
print()

print("Evaluation metrics for Random Forest:")
print("Mean Absolute Error:", mae_rf)
print("Mean Squared Error:", mse_rf)
print("R-squared (R^2):", r2_rf)


# ### denmark2 - small 'xx' & small 'yy'

# In[127]:


# Define features (independent variables) and target variable (dependent variable)
xx = denmark2.drop(columns=['economy'])  # Features
yy = denmark2['economy']  # Target variable

# Split the data into training and testing sets (75-25 split)
xx_train, xx_test, yy_train, yy_test = train_test_split(xx, yy, test_size=0.25, random_state=42)


# In[128]:


# Create a pipeline that standardizes the data then applies linear regression
pipeline = Pipeline(steps=[('scaler', StandardScaler()), ('regressor', LinearRegression())])

# Fit the pipeline to the training data
pipeline.fit(xx_train, yy_train)


# In[129]:


# Random forest regression model
random_forest_model = RandomForestRegressor(n_estimators=100, random_state=1234)

# Fit the model to the training data
random_forest_model.fit(xx_train, yy_train)


# In[130]:


# Make predictions on the testing data
yy_pred_logistic = pipeline.predict(xx_test)
yy_pred_rf = random_forest_model.predict(xx_test)


# In[131]:


# Compute evaluation metrics for logistic regression model
mae_logistic = mean_absolute_error(yy_test, yy_pred_logistic)
mse_logistic = mean_squared_error(yy_test, yy_pred_logistic)
r2_logistic  = r2_score(yy_test, yy_pred_logistic)

# Compute evaluation metrics for random forest model
mae_rf = mean_absolute_error(yy_test, yy_pred_rf)
mse_rf = mean_squared_error(yy_test, yy_pred_rf)
r2_rf  = r2_score(yy_test, yy_pred_rf)

# Print evaluation metrics
print("Evaluation metrics for Logistic Regression:")
print("Mean Absolute Error:", mae_logistic)
print("Mean Squared Error:", mse_logistic)
print("R-squared (R^2):", r2_logistic)
print()

print("Evaluation metrics for Random Forest:")
print("Mean Absolute Error:", mae_rf)
print("Mean Squared Error:", mse_rf)
print("R-squared (R^2):", r2_rf)


# ## Results

# **<U>Mean Squared Error (MSE) & Mean Absolute Error (MAE):</U> How close the model's predictions are to the actual happiness score**
# * Lower values indicate better performance
# * Lower values indicate better performance
# 
# **<U>R-Squared (R^2):</U> Indicates the proportion of variance in the happiness scores that is explained by the model**
# * A higher R-squared value indicates better model fit to the data 

# ### USA
# * **Evaluation metrics for Logistic Regression:**
#     * Mean Absolute Error: 1.385301280854837
#     * Mean Squared Error: 2.0800079468125108
#     * R-squared (R^2): 0.8971424641686121
# 
# * **Evaluation metrics for Random Forest:**
#     * Mean Absolute Error: 2.6
#     * Mean Squared Error: 7.886666666666667
#     * R-squared (R^2): 0.6100000000000001
# 
# ### FINLAND
# * **Evaluation metrics for Logistic Regression:**
#     * Mean Absolute Error: 1.0654896341077709
#     * Mean Squared Error: 2.2261193236481738
#     * R-squared (R^2): 0.8763267042417682
# 
# * **Evaluation metrics for Random Forest:**
#     * Mean Absolute Error: 2.3033333333333332
#     * Mean Squared Error: 5.8137
#     * R-squared (R^2): 0.6770166666666667
# 
# ### DENMARK
# * **Evaluation metrics for Logistic Regression:**
#     * Mean Absolute Error: 1.1684256088259053
#     * Mean Squared Error: 1.73330246016317
#     * R-squared (R^2): 0.9037054188798239
# 
# * **Evaluation metrics for Random Forest:**
#     * Mean Absolute Error: 2.2866666666666666
#     * Mean Squared Error: 6.285666666666667
#     * R-squared (R^2): 0.6507962962962963

# ### denmark2 - Feature Importance

# In[132]:


#Get feature importances
importances = random_forest_model.feature_importances_

#Create a DataFrame to view the features and their importance scores
features_xx = pd.DataFrame({'Feature': xx.columns, 'Importance': importances})

#Sort the DataFrame to see the most important features at the top
features_xx.sort_values(by='Importance', ascending=False, inplace=True)

print(features_xx)


# In[133]:


#Visualize the feature importances for Random Forest
plt.figure(figsize=(15, 30))
plt.barh(features_xx['Feature'], features_xx['Importance'], color='indigo')
plt.xlabel('Importance')
plt.gca().invert_yaxis()  
plt.show


# **<u>Feature Importance:</u> Reveals which features (predictors) have the most significant influence on predicting happiness scores**
# 
# * The Higher ranked, the better the feature

# ## Final Evaluation

# **R-Squared:** Represents the proportion of the variance in the dependent variable (GDP growth rates) that is predictable from the independent variables (features from the World Happiness Report)
# - Denmark consistently has the highest R-squared values from both Logistic Regression (0.9037) and Random Forest (0.6508) models.
#     - This indicates that the model's prediction aligns more closely with the actual GDP growth rates for Denmark compared to the USA and Finland

# **Research Question:** 
# Which country, among the USA, Finland, and Denmark, will likely experience higher GDP growth rates in 2025, using historical data from the World Happiness Report's from 2015 to 2024 while leveraging Advanced Machine Learning Algorithms for predictive analysis?
# 
# **Alternative Hypothesis (H1):** There is a significant difference in the predicted GDP growth rates among the USA, Finland, and Denmark in 2025, using historical data from the World Happiness Reports from 2015 to 2024.
# 
# **Null Hypothesis (H0):** There is no significant difference in the predicted GDP growth rates among the USA, Finland, and Denmark in 2025, using historical data from the World Happiness Reports from 2015 to 2024.
# 
# - Based on the provided data and analysis, Denmark will likely experience higher GDP growth rates in 2025 than the USA and Finland.
# 
# - Therefore, we can reject the Null Hypothesis and accept the Alternative Hypothesis, suggesting that  there is a significant difference in the predicted GDP growth rates among these countries.
# 
# 
