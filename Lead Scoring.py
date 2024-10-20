#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing required libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Importing required libraries

import warnings
warnings.filterwarnings('ignore')


# In[3]:


# Reading the data set

df = pd.read_csv("Leads.csv")
df.head(10)


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.describe()


# # Data Cleaning

# In[7]:


# Checking if there are duplicates present under Prospect ID & Lead Number.

duplicate_prospect_ID = df.duplicated(subset = 'Prospect ID')
print (sum(duplicate_prospect_ID) == 0)   


# In[8]:


duplicate_LeadNo = df.duplicated(subset = 'Lead Number')
print(sum(duplicate_LeadNo) == 0)


# There are no duplicates present in Prospect ID & Lead Number, so we can drop these columns

# In[9]:


# Removing Prospect ID & Lead Number from the data set
df.drop(['Prospect ID', 'Lead Number'],axis = 1, inplace = True)


# In[10]:


# Checking null values in every column
df.isna().sum().sort_values(ascending=False)


# In[11]:


# Converting the value 'Select' to Null
df = df.replace('Select', np.nan)


# In[12]:


# Checking null values again
df.isna().sum().sort_values(ascending=False)


# In[13]:


# Percentage of Null values in every column
round(df.isna().sum().sort_values(ascending=False)/len(df)*100, 2)


# In[14]:


# Removing the columns that have more than 45% of missing values
columns =df.columns

for i in columns:
    if((100*(df[i].isnull().sum()/len(df.index))) >= 45):
        df.drop(i, axis = 1, inplace = True)


# In[15]:


# Checking percentage of Null values again
round(df.isna().sum().sort_values(ascending=False)/len(df)*100, 2)


# Dropped columns where more than 45% of data were missing

# In[16]:


# Checking the City Column
df['City'].value_counts(dropna=False)


# In[17]:


df['City'].mode()


# In[18]:


df['City'] = df['City'].replace(np.nan,'Mumbai')


# In[19]:


# Checking Specialization column
df['Specialization'].value_counts(dropna=False)


# In[20]:


# There's a possibility that some of the customer may not have mentioned specialization as it was not in the list or there aren't any, so we can impute the 'NaN' values as 'Not Specified'

df['Specialization'] = df['Specialization'].replace(np.nan, 'Not Specified')


# In[21]:


df['Specialization'].value_counts()


# In[22]:


# Plotting the Specialization columnn 

plt.figure(figsize=(15, 5))
count_fig = sns.countplot(x='Specialization', hue='Converted', data=df, palette='nipy_spectral')
count_fig.set_xticklabels(count_fig.get_xticklabels(), rotation=90)
plt.title("Leads Conversion based on Specialization", fontsize=20)
plt.show()


# Insights- 
# 1. Management has the higher number of leads converted

# In[23]:


# Since Management is an import metric insde Specialisation, we could combine the entire Managements under one umbrella

df['Specialization'] = df['Specialization'].replace(['Finance Management','Human Resource Management',
                                                           'Marketing Management','Operations Management',
                                                           'IT Projects Management','Supply Chain Management',
                                                    'Healthcare Management','Hospitality Management',
                                                           'Retail Management'] ,'Management_Specializations')  


# In[24]:


# Plotting Specialization column after consolidating the Management Sepcializations

plt.figure(figsize=(15, 5))
count_fig = sns.countplot(x='Specialization', hue='Converted', data=df, palette='nipy_spectral')
count_fig.set_xticklabels(count_fig.get_xticklabels(), rotation=90)
plt.title("Leads Conversion based on Specialization", fontsize=20)
plt.show()


# In[25]:


# Checking percentage of Null values again
round(df.isna().sum().sort_values(ascending=False)/len(df)*100, 2)


# In[26]:


# Checking value count of column 'Tag'
df['Tags'].value_counts(dropna=False)


# Since the lead has not confirmed on the tag, we can impute Null values as "Not Specified"

# In[27]:


# Replacing 'Nan values' in Tag with 'Not Specified'
df['Tags'] = df['Tags'].replace(np.nan,'Not Specified')


# In[28]:


# Ploting on Tag variable
plt.figure(figsize=(15,5))
count_fig=sns.countplot(x='Tags', hue=df['Converted'], data = df, palette='YlGnBu')
count_fig.set_xticklabels(count_fig.get_xticklabels(),rotation=90)
plt.title("Leads Conversion based on Tags ",fontsize=20)
plt.xlabel("TAGS", fontsize=14)
plt.legend(loc=1)
plt.ylabel("Count", fontsize=14)
plt.show()


# In[29]:


#Since some tags have very minimal values, we can replace them as "Other_Tags"
df['Tags'] = df['Tags'].replace(['In confusion whether part time or DLP', 'in touch with EINS','Diploma holder (Not Eligible)',
                                     'Approached upfront','Graduation in progress','number not provided', 'opp hangup','Still Thinking',
                                    'Lost to Others','Shall take in the next coming month','Lateral student','Interested in Next batch',
                                    'Recognition issue (DEC approval)','Want to take admission but has financial problems',
                                    'University not recognized','switched off','Already a student','Not doing further education',
                                     'invalid number','wrong number given','Interested  in full time MBA'], 'Other_Tags')


# In[30]:


# Checking the column "What matters most to you in choosing a course"
df['What matters most to you in choosing a course'].value_counts(dropna=False)


# In[31]:


df['What matters most to you in choosing a course'].mode()


# In[32]:


# Replacing Null values with Mode "Better Career Prospects"

df['What matters most to you in choosing a course'] = df['What matters most to you in choosing a course'].replace(np.nan,'Better Career Prospects')


# In[33]:


# Plotting column "What matters most to you in choosing a course"
plt.figure(figsize=(15,5))
count_fig=sns.countplot(x='What matters most to you in choosing a course', hue=df['Converted'], data = df, palette='OrRd_r')
count_fig.set_xticklabels(count_fig.get_xticklabels(),rotation=0)
plt.title("Leads Conversion based on Interest ",fontsize=20)
plt.xlabel("What matters most to you in choosing a course", fontsize=14)
plt.legend(loc=1)
plt.ylabel("Count", fontsize=14)
plt.show()


# In[34]:


#checking Ratio of variable 
df['What matters most to you in choosing a course'].value_counts(dropna=False,normalize=True)*100


# Column "What matters most to you in choosing a course" does not give us an insight as the data is biases towards "Better Career Prospects", so we can drop this column

# In[35]:


df.drop('What matters most to you in choosing a course', axis=1,inplace=True)


# In[36]:


# Checking percentage of Null values again
round(df.isna().sum().sort_values(ascending=False)/len(df)*100, 2)


# In[37]:


# Checking column "What is your current occupation"

df['What is your current occupation'].value_counts(dropna=False)


# In[38]:


# Replacing Null values with "Unemployed"

df['What is your current occupation'] = df['What is your current occupation'].replace(np.nan, 'Unemployed')


# In[39]:


df['What is your current occupation'].value_counts(dropna=False)


# In[40]:


# Plotting the column "What is your current occupation"

plt.figure(figsize=(15,5))
count_fig=sns.countplot(x='What is your current occupation', hue=df['Converted'], data= df, palette='gist_earth')
count_fig.set_xticklabels(count_fig.get_xticklabels(),rotation=90)
plt.title("Leads Conversion based on Occupation",fontsize=20)
plt.show()


# In[41]:


# Ratio of categories after imputation
df['What is your current occupation'].value_counts(dropna=False,normalize = True,ascending=False)*100


# Insights-
# 1. There is a strong likelihood that working professionals will opt for the course.
# 2. The largest group among the leads consists of unemployed individuals.
# 3. Categories like housewives, businessmen, students, and others are less likely to convert and enroll in the course.

# In[42]:


# Value counts of Country column

df['Country'].value_counts()


# In[43]:


# Plotting the Country column
plt.figure(figsize=(15,5))
Count_fig=sns.countplot(x='Country', hue=df['Converted'], data = df)
Count_fig.set_xticklabels(Count_fig.get_xticklabels(),rotation=90)
plt.title("Distribution of the Count of leads across Countries")
plt.legend(loc=1)
plt.show()


# In[44]:


# Checking mode in th Country Column 
df['Country'].mode()


# In[45]:


# Replace null values with India
df['Country'] = df['Country'].replace(np.nan,'India')


# In[46]:


df['Country'].value_counts()


# In[47]:


# Visualising the Country column after replacing NaN values
plt.figure(figsize=(15,5))
Count_fig=sns.countplot(x='Country', hue=df['Converted'], data = df)
Count_fig.set_xticklabels(Count_fig.get_xticklabels(),rotation=90)
plt.title("Distribution of the Count of leads across Countries")
plt.legend(loc=1)
plt.show()


# Since "India" is tagged as the most occuring Country, it may not be suitable for an analysis - especially for a classification problem. Hence we can remove the Country column inorder to escape from the bias.

# In[48]:


# Removing Country column frmom data
df.drop('Country',axis = 1,inplace=True)


# In[49]:


# Checking percentage of Null values again
round(df.isna().sum().sort_values(ascending=False)/len(df)*100, 2)


# In[50]:


# Checking value counts of Lead Source column
df['Lead Source'].value_counts(dropna=False)


# In[51]:


# Since 'Lead Source' has less 'NaN' values we can replace it with 'Others'
df['Lead Source'] = df['Lead Source'].replace(np.nan,'Others')


# In[52]:


# Also we can combining low frequency values present in the Lead Source
df['Lead Source'] = df['Lead Source'].replace('google','Google')
df['Lead Source'] = df['Lead Source'].replace('Facebook','Social Media')
df['Lead Source'] = df['Lead Source'].replace(['bing','Click2call','Press_Release',
                                                     'youtubechannel','welearnblog_Home',
                                                     'WeLearn','blog','Pay per Click Ads',
                                                    'testone','NC_EDM'] ,'Others')                                                   


# In[53]:


# Pltting on Lead Source variable
plt.figure(figsize=(15,5))
count_fig=sns.countplot(x='Lead Source', hue=df['Converted'], data=df, palette='ocean_r')
count_fig.set_xticklabels(count_fig.get_xticklabels(),rotation=90)
plt.title("Leads Conversion based on Lead Source ",fontsize=20)
plt.xlabel("Lead Source", fontsize=14)
plt.legend(loc=1)
plt.ylabel("Count", fontsize=14)
plt.show()


# Insights-
# 1. The majority of leads are generated through Google and direct traffic, with the fewest coming from live chat.
# 2. The Welingak website has the highest conversion rate.
# 3. Improving lead conversion can be achieved by maximizing leads from references and the Welingak website.
# 4. Focusing on Olark chat, organic search, direct traffic, and Google leads could further boost lead conversion rates.

# In[54]:


# Checking 'Last Activity' variable
df['Last Activity'].value_counts(dropna=False)


# In[55]:


# Converting Null values to "others"
df['Last Activity'] = df['Last Activity'].replace(np.nan,'Others')


# In[56]:


# Replacing categories which are less than 1% to Others as it does not make much impact for analysis

df['Last Activity'] = df['Last Activity'].replace(['Unreachable','Unsubscribed',
                                                        'Had a Phone Conversation', 
                                                        'Approached upfront',
                                                        'View in browser link Clicked',       
                                                        'Email Marked Spam',                  
                                                        'Email Received','Resubscribed to emails',
                                                         'Visited Booth in Tradeshow'],'Others')


# In[57]:


df['Last Activity'].value_counts(dropna=False)


# In[58]:


# Checking percentage of Null values again
round(df.isna().sum().sort_values(ascending=False)/len(df)*100, 2)


# In[59]:


# Few columns have less than 2% na values. We can afford to drop their respective rows altogehter. 
df = df.dropna()


# In[60]:


round(df.isna().sum().sort_values(ascending=False)/len(df)*100, 2)


# In[61]:


print(len(df.index))
print(len(df.index)/9240)


# After data cleaning, we have retained 98.51% of data.

# In[62]:


# Checking the Lead Origin variable
df['Lead Origin'].value_counts(dropna=False)


# In[63]:


# Plotting count of Variable based on Converted value

plt.figure(figsize=(15,5))
count_fig=sns.countplot(x='Lead Origin', hue=df['Converted'], data = df, palette='autumn_r')
count_fig.set_xticklabels(count_fig.get_xticklabels(),rotation=0)
plt.title("Leads Conversion based on Lead Origin ",fontsize=20)
plt.xlabel("Lead Origin", fontsize=14)
plt.legend(loc=1)
plt.ylabel("Count", fontsize=14)
plt.show()


# Insights-
# 1. Both API and landing page submissions generate a high volume of leads and conversions.
# 2. While the lead add form has a strong conversion rate, the number of leads it generates is relatively low.
# 3. Increasing the number of leads through the lead add form could significantly boost the overall conversion rate and contribute to greater growth.

# In[64]:


# Checking correlation through heatmap
plt.figure(figsize=(15,5))

# Selecting only numeric columns for correlation
numeric_df = df.select_dtypes(include=['float64', 'int64'])

# Plotting the heatmap
sns.heatmap(numeric_df.corr(), cmap="Blues", annot=True)
plt.title("Heatmap on the Numerical Variables", fontsize=14)
plt.show()


# In[65]:


# Analysing the Total Time Spent on Website variable
df['Total Time Spent on Website'].describe()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




