#!/usr/bin/env python
# coding: utf-8

# In[90]:


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_percentage_error, mean_squared_error, roc_auc_score, log_loss, precision_recall_fscore_support, mean_absolute_error, plot_roc_curve
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, GridSearchCV, RandomizedSearchCV


# In[91]:


df = pd.read_csv('Bondora_raw.csv')


# In[92]:


df.head(2)


# In[93]:


df.isnull().sum()


# In[94]:


# To show all the rows of pandas dataframe
percent_missing = df.isnull().sum() * 100 / len(df)
round_percent_missing=round(percent_missing,0)
print(round_percent_missing.to_string())


# In[95]:


show_percentage_greater_than_40=(round_percent_missing>40)
print(show_percentage_greater_than_40.to_string())


# In[96]:


show_percentage_greater_than_40


# In[97]:


df.columns


# In[98]:


col_null_greater_than_40_percent=['StageActiveSince','ContractEndDate','NrOfDependants','EmploymentPosition','WorkExperience','WorkExperience','PlannedPrincipalTillDate','CurrentDebtDaysPrimary',
                                  'DebtOccuredOn','CurrentDebtDaysSecondary','DebtOccuredOnForSecondary','DefaultDate','PlannedPrincipalPostDefault','PlannedInterestPostDefault',
                                  'EAD1','EAD2','PrincipalRecovery','InterestRecovery','RecoveryStage','EL_V0','Rating_V0','EL_V1','Rating_V1',
                                  'Rating_V2','ActiveLateCategory','CreditScoreEsEquifaxRisk','CreditScoreFiAsiakasTietoRiskGrade','CreditScoreEeMini',
                                  'PrincipalWriteOffs','InterestAndPenaltyWriteOffs','PreviousEarlyRepaymentsBefoleLoan','GracePeriodStart',
                                  'GracePeriodEnd','NextPaymentDate','ReScheduledOn','PrincipalDebtServicingCost','InterestAndPenaltyDebtServicingCost','ActiveLateLastPaymentCategory']


# In[99]:


df = df.drop(col_null_greater_than_40_percent,axis=1)


# In[100]:


# To show all the rows of pandas dataframe
percent_missing = df.isnull().sum() * 100 / len(df)
round_percent_missing=round(percent_missing,0)
print(round_percent_missing.to_string())


# In[101]:


df.isnull().sum()


# In[102]:


df.shape


# In[103]:


df.dropna(axis=0,inplace=True)


# In[104]:


df.shape


# In[105]:


df.duplicated().sum()


# In[106]:


df['County'].drop


# In[107]:


df.columns


# In[130]:


features=['VerificationType','Amount','Interest','LoanDuration','MonthlyPayment','UseOfLoan','EmploymentStatus',
          'IncomeTotal','PlannedInterestTillDate','PrincipalPaymentsMade','InterestAndPenaltyBalance',
          'AmountOfPreviousLoansBeforeLoan','NrOfScheduledPayments','Status']


# In[131]:


df_new=df[features]


# In[132]:


df_new.head()


# In[133]:


#Checking distribution of categorical variables
categorical_df = df_new.select_dtypes('object')
categorical_df.info()


# In[134]:


df_new['Status'].value_counts()


# In[135]:


order_label={"Late":0,"Current":1,"Repaid":2}
df_new['Status']=df_new['Status'].map(order_label)


# In[136]:


df_new['Status'].value_counts()


# In[137]:


y = df_new['Status']
y


# In[138]:


independant_features = ['VerificationType','Amount','Interest','LoanDuration','MonthlyPayment','UseOfLoan','EmploymentStatus',
                       'IncomeTotal','PlannedInterestTillDate','PrincipalPaymentsMade',
                       'InterestAndPenaltyBalance','AmountOfPreviousLoansBeforeLoan','NrOfScheduledPayments']
X = df_new[independant_features]
X


# In[139]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)


# In[140]:


pipeline_lr=Pipeline([('scalar1',StandardScaler()),
                     ('lr_classifier',LogisticRegression(random_state=0))])


# In[141]:


pipeline_randomforest=Pipeline([('scalar3',StandardScaler()),
                     ('rf_classifier',RandomForestClassifier())])


# In[142]:


## LEts make the list of pipelines
pipelines = [pipeline_lr, pipeline_randomforest]


# In[143]:


best_accuracy=0.0
best_classifier=0
best_pipeline=""


# In[144]:


# Dictionary of pipelines and classifier types for ease of reference
pipe_dict = {0: 'Logistic Regression', 1: 'RandomForest'}

# Fit the pipelines
for pipe in pipelines:
	pipe.fit(X_train, y_train)


# In[145]:


for i,model in enumerate(pipelines):
    print("{} Test Accuracy: {}".format(pipe_dict[i],model.score(X_test,y_test)))


# In[146]:


for i,model in enumerate(pipelines):
    if model.score(X_test,y_test)>best_accuracy:
        best_accuracy=model.score(X_test,y_test)
        best_pipeline=model
        best_classifier=i
print('Classifier with best accuracy:{}'.format(pipe_dict[best_classifier]))


# In[147]:


pred = pipe.predict(X_test)
print('test accuracy = ', round(accuracy_score(y_test, pred)*100, 2), '%')


# In[148]:


print(classification_report(y_test, pred, digits=3))


# In[149]:


# save the final data
df_new.to_csv('Bondora_preprocessed.csv',index=False)


# In[150]:


df_new1=pd.read_csv('Bondora_preprocessed.csv')


# In[151]:


df_new1


# In[ ]:

import pickle
# Saving model to disk
pickle.dump(pipe, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2,9, 6,2,3,5,6,2,5,3,2,4,2]]))



# %%
