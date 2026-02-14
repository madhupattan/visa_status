#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import datetime
import warnings
warnings.filterwarnings('ignore')


# In[3]:


df=pd.read_excel('visa_status_dataset.xlsx')


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


df.duplicated().sum()


# In[7]:


df.isnull().sum()


# In[8]:


df['application_date']=pd.to_datetime(df['application_date'])


# In[9]:


df['decision_date']=pd.to_datetime(df['decision_date'])


# In[10]:


df['processing_days']=(df['decision_date']-df['application_date']).dt.days


# In[11]:


#Average processing time

print("Average processing time:", df["processing_days"].mean())


# In[12]:


#create a risk column

df["risk"] = df["processing_days"].apply(
    lambda x: "High Delay" if x > 30 else "Low Delay"
)


# In[13]:


df.head()


# In[14]:


# Numpy for calculations

processing_array = np.array(df["processing_days"])

print("Mean:", np.mean(processing_array))
print("Median:", np.median(processing_array))
print("Max:", np.max(processing_array))
print("Min:", np.min(processing_array))
print("Std Dev:", np.std(processing_array))


# In[15]:


# Normalisation

df["normalized_processing"] = (
    (df["processing_days"] - np.mean(df["processing_days"])) /
    np.std(df["processing_days"])
)


# In[16]:


# Preprocessing

# Filling missing values

df['visa_status'].fillna('Unknown',inplace=True)
df['nationality'].fillna('Unknown',inplace=True)
df['processing_center'].fillna('Unknown',inplace=True)
df['applicant_age']=df['applicant_age'].fillna(df['applicant_age'].mean()).round().astype(int)
df['gender'].fillna(df['gender'].mode()[0],inplace=True)


# In[17]:


# Preparing data for Model Building

df=pd.get_dummies(df,drop_first=True)


# In[18]:


df.head()


# In[ ]:




