
# coding: utf-8

# In[1]:


# importing statements
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Reading datasets
train = pd.read_csv("F:\Project\Loan Approval\\train.csv")
test = pd.read_csv("F:\Project\Loan Approval\\test.csv")

# Creating a copy of datasets
train_orig = train.copy()
test_orig = test.copy()

# Viewing the datasets
train.columns

test.columns

train.dtypes

test.dtypes

train.shape

test.shape


# In[ ]:


# Univariate Analysis (Analysing each feature variable)
# target variable
train['Loan_Status'].value_counts()

train['Loan_Status'].value_counts(normalize=True)


# In[ ]:


import matplotlib.pyplot as plt

train['Loan_Status'].value_counts().plot.bar()


# In[ ]:


# Univariate Analysis (Categorical Variables: Married, Gender, Self_Employed, Credit_History)
import matplotlib.pyplot as plt

plt.figure(1)
plt.subplot(221)
train['Married'].value_counts().plot.bar(figsize = (20,10), title = "Married")

plt.subplot(222)
train['Gender'].value_counts().plot.bar(title = "Gender")

plt.subplot(223)
train['Self_Employed'].value_counts().plot.bar(title = "Self+Employed")

plt.subplot(224)
train['Credit_History'].value_counts().plot.bar(title = "Credit_history")


# In[ ]:


# Univariate Analysis (ordinal Variables : Dependents, Education, Property_Area)
import matplotlib.pyplot as plt

plt.figure(1)
plt.subplot(311)
train['Education'].value_counts().plot.bar(figsize=(10,10),title = "Education")

plt.subplot(312)
train['Dependents'].value_counts().plot.bar(title = "Dependents")

plt.subplot(313)
train['Property_Area'].value_counts().plot.bar(title= "property_Area")


# In[ ]:


# Univariate Analysis : Numerical variables (ApplicantIncome)
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(1)
plt.subplot(121)
sns.distplot(train['ApplicantIncome']);

plt.subplot(122)
train['ApplicantIncome'].plot.box(title = "ApplicantIncome")


# In[ ]:


# Univariant Analysis : Numerical variables (CoapplicantIncome)
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(1)
plt.subplot(121)
sns.distplot(train['CoapplicantIncome']);

plt.subplot(122)
train['CoapplicantIncome'].plot.box(title = "CoapplicantIncome", figsize=(15,5))


# In[ ]:


# Bivariate Analysis : categorical variables vs target variable (gender vs Loan_Status, Married vs Loan_Status,
# Dependents vs Loan_Status, Education vs Loan_Status, Self_Employed vs Loan_Status, Credit_History vs Loan_Status,
# property_Area vs Loan_Status)

import matplotlib.pyplot as plt

Gender = pd.crosstab(train['Gender'],train['Loan_Status'])
Married = pd.crosstab(train['Married'], train['Loan_Status'])
Dependents = pd.crosstab(train['Dependents'], train['Loan_Status'])
Education = pd.crosstab(train['Education'],train['Loan_Status'])
Self_Employed = pd.crosstab(train['Self_Employed'],train['Loan_Status'])
Credit_History= pd.crosstab(train['Credit_History'],train['Loan_Status'])
Property_Area = pd.crosstab(train['Property_Area'], train['Loan_Status'])

Gender.div(Gender.sum(1).astype(float), axis = 0).plot.bar(stacked = True, figsize = (5,4), title = "Gender vs Loan_Status")
plt.show()

Married.div(Married.sum(1).astype(float), axis =0).plot.bar(stacked = True, figsize = (5,4), title = "Married vs Loan_Status")
plt.show()

Dependents.div(Dependents.sum(1).astype(float), axis = 0).plot.bar(stacked = True, figsize = (5,4), title = "Dependents vs Loan_Status")
plt.show()

Education.div(Education.sum(1).astype(float), axis = 0).plot.bar(stacked = True, figsize = (5,4), title = "Education vs Loan_status")
plt.show()

Self_Employed.div(Self_Employed.sum(1).astype(float),axis = 0).plot.bar(stacked = True, figsize =(5,4), title = "Self_Employed vs Loan_Status")
plt.show()

Credit_History.div(Credit_History.sum(1).astype(float),axis =0).plot.bar(stacked = True, figsize = (5,4), title = "Credit_History vs Loan_Status")
plt.show()

Property_Area.div(Property_Area.sum(1).astype(float),axis=0).plot.bar(stacked = True, figsize =(5,4),title = "Property_Area vs Loan_Status")
plt.show()


# In[ ]:


# Bivariant Analysis: Numerical variables vs target variable

import matplotlib.pyplot as plt

train.groupby('Loan_Status')['ApplicantIncome'].mean().plot.bar()


# In[ ]:


import pandas as pd
bins = [0,2500,4000,6000,81000]
group = ['low','average', 'high','veryhigh']
train['Income_bin'] = pd.cut(train['ApplicantIncome'], bins, labels = group)

Income_bin = pd.crosstab(train['Income_bin'],train['Loan_Status'])
Income_bin.div(Income_bin.sum(1).astype(float),axis=0).plot.bar(stacked = True, figsize=(5,5))
plt.xlabel('ApplicantIncome')
P = plt.ylabel('Percentage')


# In[ ]:


# Bivariant analysis : Coapplicant_Income variable vs Loan_Status

import matplotlib.pyplot as plt

bins=[0,1000,3000,42000]
group=['Low','Average','High']
train['Coapplicant_Income_bin']=pd.cut(train['CoapplicantIncome'],bins,labels=group)

Coapplicant_Income_bin = pd.crosstab(train['Coapplicant_Income_bin'],train['Loan_Status'])
Coapplicant_Income_bin.div(Coapplicant_Income_bin.sum(1).astype(float),axis=0).plot.bar(stacked = True)
plt.xlabel=('Coapplicant_Income')
P=plt.ylabel('percentage')


# In[ ]:


# Bivariant Analysis : Calculated TotalIncome and TotalIncome vs Loan_Status

import matplotlib.pyplot as plt

train['Total_Income'] = train['ApplicantIncome']+train['CoapplicantIncome']

bins=[0,2500,4000,6000,81000]
group=['Low','Average','High', 'Very high']
train['Total_Income_bin']=pd.cut(train['Total_Income'],bins,labels=group)

Total_Income_bin=pd.crosstab(train['Total_Income_bin'],train['Loan_Status'])
Total_Income_bin.div(Total_Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('Total_Income')
P = plt.ylabel('Percentage')


# In[ ]:


# Bivariant Analysis : LoanAmount and Loan_Status

import matplotlib.pyplot as plt
import pandas as pd

bins=[0,100,200,700]
group=['Low','Average','High']
train['LoanAmount_bin']=pd.cut(train['LoanAmount'],bins,labels=group)

LoanAmount_bin=pd.crosstab(train['LoanAmount_bin'],train['Loan_Status'])
LoanAmount_bin.div(LoanAmount_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('LoanAmount')
P = plt.ylabel('Percentage')


# In[ ]:


# lets drop the bins

train=train.drop(['Income_bin', 'Coapplicant_Income_bin', 'LoanAmount_bin', 'Total_Income_bin', 'Total_Income'], axis=1)

