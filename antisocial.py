#The complete file of the problem turn_anti_social

#import dependencies
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import scipy.stats as st
import statsmodels.api as sm

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import catboost as cb
import xgboost as xgb
import lightgbm as lgbm

plt.rcParams['figure.figsize']= (16, 12)
plt.style.use('fivethirtyeight')

sns.set_color_codes(palette= 'muted')
sns.set(color_codes= True, style= 'ticks')

pd.options.display.max_rows= 1000
pd.options.display.max_columns= 199

#Read the data
train= pd.read_csv('C://Users//sngupta//Documents//hackerearthhackathon//zs_hiring_challenge//DS_Anti_Social_v1//DS_SOCIAL_NETWORK_TEST//train.csv')
test= pd.read_csv('C://Users//sngupta//Documents//hackerearthhackathon//zs_hiring_challenge//DS_Anti_Social_v1//DS_SOCIAL_NETWORK_TEST//test.csv')
sample= pd.read_csv('C://Users//sngupta//Documents//hackerearthhackathon//zs_hiring_challenge//DS_Anti_Social_v1//DS_SOCIAL_NETWORK_TEST//sample_submission.csv')

target= train['turn_anti_social'] #Extracted the target values in ne variable

#Do some analysis of the dataset given
#The glimpse of the training data
print(train.head())

#Check out the target from that I conclude is dataset imbalanced or not
train['turn_anti_social'].value_counts().plot(kind= 'bar', figsize= (4, 4))
plt.title('Say No to Social Media or Not')
plt.show()

#There is very less number which turn on anti_social
#It is an imbalanced class problem because the difference between the positive and negative result is very
#For the better result we have to maintain the class weight or have to do some manipulation of the data

#The information of the dataset
print(train.info())

#the discription of the dataset
print(train.describe())

#the unique elements present in particular features
print(train.nunique())

#Here, I plotted the correlation graph between the features given
plt.figure(figsize= (16, 16))
sns.heatmap(train.corr(), annot= True, fmt= '.2f', square= True)
plt.xticks(size= 12)
plt.yticks(size= 12)
plt.show()

#From the correlation we draw that the charges of different platform depends on the character used
#Now here, we can fill the null values of the charges with the number of character used

#Check out the null values present in the dataset
print('The null values present in the training dataset: ')
print(train.isnull().sum().sort_values(ascending= False)[:10])

print('The null values present in the testing dataset: ')
print(test.isnull().sum().sort_values(ascending= False)[:10])

#After correlation matrix I'm going to plot pairplot between the charges and number of msgs 
sns.pairplot(train[['total_facebook_charge', 'total_twitter_charge', 'total_whatsapp_msg_characters',
                      'total_email_charge']], palette= 'muted')
plt.show()

#Lets look the joint plot between the charge or characters and the number of statuses or tweets or msgs
sns.jointplot('total_facebook_statuses', 'total_facebook_charge', data= train)
sns.jointplot('total_twitter_tweets', 'total_twitter_charge', data= train)
sns.jointplot('total_whatsapp_msgs', 'total_whatsapp_msg_characters', data= train)
sns.jointplot('total_emails', 'total_email_charge', data= train)

#Box-plot between some features and the target faeture
cols_plot= ['total_facebook_statuses', 'number_of_snaps', 'total_twitter_tweets', 'total_whatsapp_msgs', 'total_emails',
            'account_membership_period']
for i, col in enumerate(cols_plot):
    plt.subplot(3,2,i+1)
    sns.boxplot('turn_anti_social', col, data= train, palette= 'deep')
    plt.title(str(col) + ' and Turn Anti Social')
plt.tight_layout()
plt.show()

#Data cleaning
def clean_data(df_train, df_test): #Basically, this function used to clean the dataset means drop missing values, etc
    #The shape of dataset before cleaning
    print('The shape of the train dataset: {}' .format(df_train.shape))
    print('The shape of the test dataset: {}' .format(df_test.shape))
    
    #Now start
    #from above analysis, you simply keep either the charges or the number of characters on conataining null values 
    #because from the correlation graph we found that both have correlation matrix 1.0
    #Thus, now we can drop the columns which contains the null values either charges or the characters used
    cols_to_drop= ['total_email_characters', 'total_facebook_status_characters', 'total_twitter_tweet_characters', 
                   'total_whatsapp_charge']

    #Now the new dataframe I'm going to create is like that
    df_train= df_train.drop(cols_to_drop, axis= 1)
    df_test= df_test.drop(cols_to_drop, axis= 1)
    
    #Now the question arises what to do with the social_account_number, have to keep or discard it
    #lets do some analysis
    #As we saw the nunique elements each and every element of social_account_number feature is unique
    #Thus, we say its act like a unique_id and generally we don't use any unique_id in modeling
    #Hence, we have to discard it
    df_train.drop(['social_account_number', 'uid', 'turn_anti_social'], axis= 1, inplace= True)
    df_test.drop(['social_account_number', 'uid'], axis= 1, inplace= True)
    
    df_all= pd.concat([df_train, df_test])
    #Here, some of the features are of object type thus we need to covert it into numerical form
    encoder= LabelEncoder()
    
    df_all['country']= encoder.fit_transform(df_all['country'].astype(str))
    
    df_train= df_all[:len(df_train)]
    df_test= df_all[len(df_train):]
    
    df_train['email_plan']= df_train['email_plan'].map({'yes': 1, 'no': 0})
    df_test['email_plan']= df_test['email_plan'].map({'yes': 1, 'no': 0})
    df_train['snapchat_plan']= df_train['snapchat_plan'].map({'yes': 1, 'no': 0})
    df_test['snapchat_plan']= df_test['snapchat_plan'].map({'yes': 1, 'no': 0})
    
    print('The shape of the training dataset after cleaning: {}' .format(df_train.shape))
    print('The shape of the testing dataset after cleaning: {}' .format(df_test.shape))
    
    return df_train, df_test
 
train_1, test_1= clean_data(train, test)
#Now the time to modeling
#Basically, I got the best result using the XGBoost Classifier
def run_XGB(train, target, test):
    
    param_xgb= {}
    param_xgb['eda']= 0.1
    param_xgb['max_depth']= 3
    param_xgb['objective']= 'binary:logistic'
    param_xgb['eval_metric']= 'auc'
    param_xgb['seed']= 2018
    param_xgb['max_delta_step']= 77
    
    #split the data in trainig and validating form
    X_train, X_test, y_train, y_test= train_test_split(train, target, test_size= 0.2, random_state= 2018)
    #prepare data for the training
    dtrain= xgb.DMatrix(X_train, y_train)
    dtest= xgb.DMatrix(X_test, y_test)
    dtest_1= xgb.DMatrix(test)
    
    watchlist= [(dtrain, 'train'), (dtest, 'eval')]
    
    model= xgb.train(param_xgb, dtrain, num_boost_round= 1000, evals= watchlist, verbose_eval= 100, early_stopping_rounds= 50)
    
    #do prediction
    prediction= model.predict(dtest)
    #prediction_1= prediction.copy()

    prediction_2= model.predict(dtest_1)
    
    for i in range(len(prediction)):
        if prediction[i]<0.5:
            prediction[i]= 0
        else:
            prediction[i]= 1
            
    for i in range(len(prediction_2)):
        if prediction_2[i]<=0.5:
            prediction_2[i]= 0
        else:
            prediction_2[i]= 1
            
    
    #accuracy
    print('Testing accuracy: {:.4f}' .format(accuracy_score(y_test, prediction)))
    print('f1 score: {}' .format(f1_score(y_test, prediction)))
    print('ROC AUC: {}' .format(roc_auc_score(y_test, prediction)))
    
    print('Confusion Matrix: ')
    cm= confusion_matrix(y_test, prediction)
    print(cm)
    print('Classification Report: ')
    print(classification_report(y_test, prediction))
    
    TP = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TN = cm[1][1]

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)

    print('True Positive Rate: {}' .format(TPR))
    print('True Negative Rate: {}' .format(TNR))
    print('False Positive Rate: {}' .format(FPR))
    print('False Negative Rate: {}' .format(FNR))
    print('Overall Accuracy: {}' .format(ACC))
    
    #plot the feature Importance
    xgb.plot_importance(model)
    plt.show()
    
    return prediction_2

prediction= run_XGB(train_1, target, test_1)
#Now create sample dataframe
sample['turn_anti_social']= prediction.astype(int)
sample.to_csv('C://Users//sngupta//Documents//hackerearthhackathon//zs_hiring_challenge//DS_Anti_Social_v1//sample.csv', index= False)

    
