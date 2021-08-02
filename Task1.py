import pandas as pd
from sklearn.metrics import  accuracy_score
import sklearn
import random
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

pd.set_option('max_columns',None)
path_train=r'.\train.tsv'
path_test=r'.\test.tsv'
data_train=pd.read_csv(path_train,sep='\t').head(60000)
data_test=pd.read_csv(path_test,sep='\t').head(10000)
index_seq=list(range(1,len(data_train)))
random.seed(6)
random.shuffle(index_seq)
index_train=index_seq[:int(len(index_seq)*0.75)]
index_val=index_seq[int(len(index_seq)*0.75):]
#获取句子

cv=CountVectorizer()
x_train=data_train['Phrase'].values.tolist()
x_train=cv.fit_transform(x_train).toarray()
x_val=[x_train[index] for index in index_val]
x_train=[x_train[index] for index in index_train]
y_train=data_train['Sentiment'].values.tolist()
y_val=[y_train[index] for index in index_val]
y_train=[y_train[index] for index in index_train]
#classifier
classifier=LogisticRegression(solver='newton-cg')
classifier.fit(x_train,y_train)
pred=classifier.predict(x_val)
#输出测试集结果
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
print(accuracy_score(y_val,pred))
print(f1_score(y_val, pred, average='weighted'))