#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DATA SCIENCES QUIZZES

QUIZ 3
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import itertools

#%% LOAD DATA
adult_data = pd.read_csv("./Data/adult_data.csv", header=None, sep=",")
adult_test = pd.read_csv("./Data/adult_test.csv", header=None, sep=",")

col_names = ['Age','Workclass','fnlgwt','Education','Education-Num','Marital-Status',
           'Occupation','Relationship','Race','Sex','Capital-Gain','Capital-Loss',
           'Hours-per-Week','Country','Income']


adult_data.columns = col_names
adult_test.columns = col_names

#adult_data.iloc[0]
#adult_data.describe()
adult_data.shape

#%% ANALYZE
# How many missing values?
num_data = adult_data.shape[0]
for c in col_names:
    num_non = adult_data[c].isin([' ?']).sum()
    if num_non > 0:
        print (c)
        print (num_non)
        print ("{0:.2f}%".format(float(num_non) / num_data * 100))
        print ("\n")


# Replace missing values with nan and then remove the rows
for i in adult_data.columns:
    adult_data[i].replace(' ?', np.nan, inplace=True)
    adult_test[i].replace(' ?', np.nan, inplace=True)


adult_data.dropna(how='any',inplace=True)
adult_test.dropna(how='any',inplace=True)
adult_data.shape


# SUMMARY STATISTICS
# Value_counts for categorical attributes
adult_data["Workclass"].value_counts()
adult_data["Education"].value_counts()
adult_data["Marital-Status"].value_counts()
adult_data["Occupation"].value_counts()
adult_data["Relationship"].value_counts()
adult_data["Race"].value_counts()
adult_data["Sex"].value_counts()
adult_data["Country"].value_counts()

# Reduce Country in United-States - Other --> PERCENTUALE DI US?
adult_data.loc[adult_data['Country'] != ' United-States', 'Country'] = ' Other'
adult_test.loc[adult_test['Country'] != ' United-States', 'Country'] = ' Other'
#dataset.loc[dataset['country'] == ' United-States', 'country'] = 'US'
#adult_data.Country.value_counts().plot(kind='bar') #histogram Us - non Us

# Reduce Race in White - Other --> PERCENTUALE DI WHITE?
adult_data.loc[adult_data['Race'] != ' White', 'Race'] = ' Other'
adult_test.loc[adult_test['Race'] != ' White', 'Race'] = ' Other'



# Binarize Income
adult_data.loc[adult_data['Income'] == ' >50K', 'Income'] = 1
adult_data.loc[adult_data['Income'] == ' <=50K', 'Income'] = 0

# CONFRONT Education with Education-Num
#adult_data.drop(labels='Education', axis=1, inplace=True)
#adult_test.drop(labels='Education', axis=1, inplace=True)

#%% VISULIZATION
adult_data.Age.plot.hist(bins=18)
plt.show()

# ALL HISTOGRAMS
fig = plt.figure(figsize=(20,15))
cols = 5
rows = 3
for i, column in enumerate(adult_data.columns):
    ax = fig.add_subplot(rows, cols, i + 1)
    ax.set_title(column)
    if adult_data.dtypes[column] == np.object:
        adult_data[column].value_counts().plot(kind="bar", axes=ax)
    else:
        adult_data[column].hist(axes=ax)
        plt.xticks(rotation="vertical")
plt.subplots_adjust(hspace=0.7, wspace=0.2)

#%% CORRELATION between non categorical variables
corr= adult_data.corr()
fig, ax =plt.subplots(figsize=(15, 15))
ax.matshow(corr)
plt.xticks(range(len(corr.columns)),corr.columns)
plt.yticks(range(len(corr.columns)),corr.columns)
plt.show()


sns.heatmap(adult_data.corr())
plt.show()

# Calculate the correlation and plot it

sns.heatmap(adult_data.corr(), square=True)
plt.show()


#%% FIRST FINDING
#test_1 = adult_data[['Education','Education-Num']]
#test_1.sort_values(by='Education-Num', inplace=True)


total_elements = []
high_income = []
labels = []

education_levels = adult_data.Education.unique()
for i in education_levels:
    tmp = adult_data.loc[adult_data['Education'] == i]
    high_income.append(tmp.Income.sum())
    total_elements.append(tmp.shape[0])
    labels.append(i)
    
proportion = np.divide(high_income,total_elements)
indexes = np.argsort(proportion)
proportion = proportion[indexes]
labels = [labels[i] for i in indexes]



barra_x_test = np.arange(len(proportion))

plt.bar(barra_x_test, proportion, align='center')
plt.xticks(barra_x_test, labels, rotation='vertical')
plt.show()
#encoded_test_1, _ = number_encode_features(test_1)
#correlation = encoded_test_1.corr()
#print(correlation)

#%% SECOND FINDING
# Plot percentage of occupation per income class
total_elements = []
high_income = []
labels = []

education_levels = adult_data['Marital-Status'].unique()
for i in education_levels:
    tmp = adult_data.loc[adult_data['Marital-Status'] == i]
    high_income.append(tmp.Income.sum())
    total_elements.append(tmp.shape[0])
    labels.append(i)
    
proportion = np.divide(high_income,total_elements)
indexes = np.argsort(proportion)
proportion = proportion[indexes]
labels = [labels[i] for i in indexes]



barra_x_test = np.arange(len(proportion))

plt.bar(barra_x_test, proportion, align='center')
plt.xticks(barra_x_test, labels, rotation='vertical')
plt.show()



#%% TRANSFORM DATA
def number_encode_features(df):
    result = df.copy()
    encoders = {}
    for column in result.columns:
        if result.dtypes[column] == np.object:
            encoders[column] = preprocessing.LabelEncoder()
            result[column] = encoders[column].fit_transform(result[column])
    return result, encoders


encoded_data, _ = number_encode_features(adult_data)
encoded_test, _ = number_encode_features(adult_test)



#%% LEARN FIT
X_train = encoded_data[encoded_data.columns.drop('Income', 'Education')]
y_train = encoded_data['Income']

X_test = encoded_test[encoded_test.columns.drop('Income', 'Education')]
y_test = encoded_test['Income']

scaler = preprocessing.StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scaler.fit_transform(X_test), columns=X_train.columns)

cls = LogisticRegression()
cls.fit(X_train, y_train)

#%% PREDICT
y_pred = cls.predict(X_test)

#%% ASSES
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
accuracy_score = metrics.accuracy_score(y_test, y_pred)
cm = metrics.confusion_matrix(y_test, y_pred)
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cm, classes= ['0', '1'],
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cm, classes=['0', '1'], normalize=True,
                      title='Normalized confusion matrix')

plt.show()

print(accuracy_score)



