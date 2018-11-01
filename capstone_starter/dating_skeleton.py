import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

all_data = pd.read_csv("profiles.csv")

# *****************************************************************************
# Drafts
print('*'*80)
print('DATA SET PREVIEW')
print(all_data.head())
print('*'*80)
print('ALL COLUMNS')
print(all_data.columns)
#print(all_data.body_type.head())
#print('*'*80)

# *****************************************************************************
# Exploration of the dataset
# Interested in:
#               body_type
#               age
#               income

# body_type
print('*'*80)
print('\t BODY TYPE')
print('*'*80)
print('*'*40)
print('\t first few line')
print('*'*40)
print(all_data.body_type.head(10))
print('*'*40)
print('\t summary')
print('*'*40)
print(all_data["body_type"].value_counts())
# education
print('*'*80)
print('\t AGE')
print('*'*80)
print('*'*40)
print('\t first few line')
print('*'*40)
print(all_data.age.head(10))
print('*'*40)
print('\t summary')
print('*'*40)
print(all_data.age.value_counts())
# income
print('*'*80)
print('\t INCOME')
print('*'*80)
print('*'*40)
print('\t first few line')
print('*'*40)
print(all_data.income.head(10))
print('*'*40)
print('\t summary')
print('*'*40)
print(all_data.income.value_counts())

# *****************************************************************************
# visualization
# Interested in:
#               body_type
#               age
#               income

# body_type
#   summary
temp = all_data.body_type.value_counts()
print(temp.index)
plt.figure(figsize=(20,10))
plt.bar(temp.index,temp.values)
plt.xlabel('body type')
plt.ylabel('counts')
plt.savefig('body_type_count.png',  dpi=200)


# age
#   summary

plt.figure(figsize=(20,10))
plt.hist(all_data.age, bins = 20)
plt.xlabel('age')
plt.ylabel('counts')
plt.xlim(10,80 )
plt.savefig('age_hist.png',  dpi=200)

# income
#   summary
temp = all_data.income.value_counts()
print(temp.index)
plt.figure(figsize=(20,10))
plt.bar([str(x) for x in temp.index],temp.values)
plt.xlabel('income')
plt.ylabel('counts')
plt.savefig('income_count.png',  dpi=200)

# *****************************************************************************
# FORMULATING QUESTION
# QUESTION:
#           Classify pets using body_type and education
print('*'*80)
print('\t PETS')
print('*'*80)
print(all_data.pets.value_counts())

# *****************************************************************************
# AUGMENTING DATA
body_type_mapping = {'average':0,'fit':1,'athletic':2,'thin':3,'curvy':4,'a little extra':5,'skinny':6,'full figured':7,'overweight':8,'jacked':9,'used up':10,'rather not say':11}
all_data['body_type_code'] = all_data.body_type.map(body_type_mapping)
pets_mapping = {'likes dogs and likes cats':2,'likes dogs':1,'likes dogs and has cats':2,'has dogs':1,'has dogs and likes cats':2,'likes dogs and dislikes cats':1,'has dogs and has cats':2,'has cats':1,'likes cats':1,'has dogs and dislikes cats':1,'dislikes dogs and likes cats':1,'dislikes dogs and dislikes cats':0,'dislikes cats':0,'dislikes dogs and has cats':1,'dislikes dogs':1}
all_data['pets_code'] = all_data.pets.map(pets_mapping)

# *****************************************************************************
# NORMALIZE DATA
feature_data = all_data[['age','body_type_code']]
feature_data.fillna({'body_type_code':11}, inplace = True)
x=feature_data.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
feature_data =  pd.DataFrame(x_scaled,columns=feature_data.columns)
#feature_data.isna().any()
feature_label = all_data[['pets_code']]
feature_label.fillna({'pets_code':0},inplace=True)
#feature_label.isna().any()

# *****************************************************************************
# Split the set
x_train, x_test, y_train, y_test = train_test_split(feature_data, feature_label, train_size = 0.8, test_size = 0.2, random_state = 10)

# *****************************************************************************
def regression_predict(df):
    for i in range(len(df)):
        ele=df[i]
        if ele<2./3:
            df[i]=0
        elif ele<4./3:
            df[i]=1
        else:
            df[i]=2
    return df
def compute_accuracy(df1,df2):
    cnt, tot = 0, 0.
    for ele1,ele2 in zip(df1, df2):
        if ele1 == ele2:
            cnt += 1
        tot += 1
    return cnt/tot
# Linear regression
multi_linear_regression = LinearRegression()
multi_linear_regression.fit(x_train, y_train)
y_predict_mlr = regression_predict(multi_linear_regression.predict(x_test))

plt.figure(figsize=(10,10))
plt.scatter(y_test, y_predict_mlr,alpha=0.01)
plt.xlabel('y test')
plt.ylabel('y predict')
plt.savefig('multilinear regression.png')

print('*'*40)
print('Accuracy of multilinear regression:%f'%compute_accuracy(y_test.values, y_predict_mlr))
print('*'*40)
# *****************************************************************************
# K neighborhood regressor
from sklearn.neighbors import KNeighborsRegressor
KN_regression = KNeighborsRegressor(n_neighbors=3,weights='distance')
KN_regression.fit(x_train,y_train)
y_predict_knr = regression_predict(KN_regression.predict(x_test))

plt.figure(figsize=(10,10))
plt.scatter(y_test, y_predict_knr,alpha=0.01)
plt.xlabel('y test')
plt.ylabel('y predict')
plt.savefig('K neighbors regression.png')

print('*'*40)
print('Accuracy of K neighbors regression:%f'%compute_accuracy(y_test.values,y_predict_knr))
print('*'*40)

# *****************************************************************************
# KNN
from sklearn.neighbors import KNeighborsClassifier
KN_classifier = KNeighborsClassifier(n_neighbors = 3)
KN_classifier.fit(x_train, y_train)
y_predict_knn = KN_classifier.predict(x_test)

plt.figure(figsize=(10,10))
plt.scatter(y_test, y_predict_knn,alpha=0.01)
plt.xlabel('y test')
plt.ylabel('y predict')
plt.savefig('K neighbors classifier.png')

print('*'*40)
print('Accuracy of K neighbors classifier:%f'%compute_accuracy(y_test.values,y_predict_knn))
print('*'*40)

# *****************************************************************************
# SVM
from sklearn.svm import SVC
def is_i(df,i):
    res = []
    for ele in df.values:
        if ele==i:
            res.append(1)
        else:
            res.append(0)
    return res
def combine_01(df0,df1):
    res = []
    for ele0,ele1 in zip(df0,df1):
        if ele0:
            res.append(0)
        elif ele1:
            res.append(1)
        else:
            res.append(2)
    return res
y_train0 = is_i(y_train,0)
y_train1 = is_i(y_train,1)
svm_classifier0 = SVC(kernel='rbf')
svm_classifier1 = SVC(kernel='rbf')
svm_classifier0.fit(x_train,y_train0)
svm_classifier1.fit(x_train,y_train1)

y_predict_svm0= svm_classifier0.predict(x_test)
y_predict_svm1= svm_classifier1.predict(x_test)

y_predict_svm = combine_01(y_predict_svm0, y_predict_svm1)

plt.figure(figsize=(10,10))
plt.scatter(y_test, y_predict_svm,alpha=0.01)
plt.xlabel('y test')
plt.ylabel('y predict')
plt.savefig('SVM.png')

print('*'*40)
print('Accuracy of SVM:%f'%compute_accuracy(y_test.values,y_predict_svm))
print('*'*40)

    
plt.show()