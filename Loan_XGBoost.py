
# coding: utf-8

# In[9]:


# importing statements
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost.sklearn import XGBClassifier
# Reading datasets
train = pd.read_csv("F:\Project\Loan Approval\\train.csv")
test = pd.read_csv("F:\Project\Loan Approval\\test.csv")

# Creating a copy of datasets
train_orig = train.copy()
test_orig = test.copy()

# lets replace 3+ in dependents to 3
#train['Dependents'] = map(lambda x: x.replace("3+","3"), train['Dependents'])
train['Dependents'].replace({ '3+' : 3},inplace=True)
test['Dependents'].replace({'3+':3},inplace = True)

# lets convert Loan_Status to numericals
train['Loan_Status'].replace('N',0,inplace = True)
train['Loan_Status'].replace('Y',1,inplace = True)

# Imputing missed values with mode in categorical variables : train data
train['Gender'].fillna(train['Gender'].mode()[0],inplace = True)
train['Married'].fillna(train['Married'].mode()[0], inplace=True)
train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)

# Imputing missed values with mode in categorical variables : test data
test['Gender'].fillna(test['Gender'].mode()[0],inplace = True)
test['Married'].fillna(test['Married'].mode()[0], inplace=True)
test['Dependents'].fillna(test['Dependents'].mode()[0], inplace=True)
test['Self_Employed'].fillna(test['Self_Employed'].mode()[0], inplace=True)
test['Credit_History'].fillna(test['Credit_History'].mode()[0], inplace=True)

# Imputing numerical variables : test data
# Imputing Loan_Amount_Term with mode
train['Loan_Amount_Term'].fillna(test['Loan_Amount_Term'].mode()[0],inplace = True)

# Imputing numerical variables : test data
# Imputing Loan_Amount_Term with mode
test['Loan_Amount_Term'].fillna(test['Loan_Amount_Term'].mode()[0],inplace = True)

#imputing LoanAmount with median : train data
train['LoanAmount'].fillna(train['LoanAmount'].median(),inplace = True)

#imputing LoanAmount with median : test data
test['LoanAmount'].fillna(test['LoanAmount'].median(),inplace = True)

# Outliers treatment
train['LoanAmount_log'] = np.log(train['LoanAmount'])
#train['LoanAmount_log'].hist(bins = 20)
test['LoanAmount_log'] = np.log(test['LoanAmount'])

train['Total_Income'] = train['ApplicantIncome']+train['CoapplicantIncome']
test['Total_Income'] = test['ApplicantIncome'] + test['CoapplicantIncome']

train=train.drop(['ApplicantIncome','CoapplicantIncome'], axis =1)
test = test.drop(['ApplicantIncome','CoapplicantIncome'],axis = 1)

# Dropping Loan_ID variable which is not useful in this context
train= train.drop('Loan_ID', axis=1)
test= test.drop('Loan_ID',axis=1)

# Defining X and Y labels for our train data
X = train.drop('Loan_Status',1)
y = train.Loan_Status

# We will use Logistic Regression, which only requires numerical values.
# Gender doesnt contain numerics, so by using dummies we will convert it to numerics.
X = pd.get_dummies(X)
train= pd.get_dummies(train)
test = pd.get_dummies(test)

# we will use train_test_split function from sklearn to split train data
x_train,x_cv,y_train,y_cv=train_test_split(X,y,test_size = 0.3)


# Modeling Logistic Regression
model = XGBClassifier(n_estimators=50, max_depth=4)
model.fit(x_train,y_train)

# Prediction
pred_cv = model.predict(x_cv)

# Accuracy score
score = accuracy_score(y_cv,pred_cv)
print('accuracy_score:',score) 

# Predictions for test dataset
pred_test = model.predict(test)

# Importing the submission file
submission = pd.read_csv("F:\Project\Loan Approval\\Sample_Submission.csv")

# As we need only loan_Id and Loan_status, we need to load them into Submission dataframe
submission['Loan_Status'] = pred_test
submission['Loan_ID'] = test_orig['Loan_ID']

# we need predictions in Y and N, so we convert 0 and 1 to N and Y respectively
submission['Loan_Status'].replace(0,'N', inplace =True)
submission['Loan_Status'].replace(1,'Y', inplace=True)

# covert the submission dataframe to .csv file
pd.DataFrame(submission, columns = ['Loan_ID','Loan_Status']).to_csv('loan_xgboost.csv',index=False)

