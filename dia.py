
import pandas as pd
import numpy as np
import itertools
import time
import sys
import os
import math
#from sklearn import datasets
from collections import Counter
import matplotlib.pyplot as plt
import datetime

print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
start = datetime.datetime.now()

os.chdir('E:/Fall2020/MachineL')

csv_data = pd.read_csv('E:/Fall2020/MachineL/diabetes.csv')  # 读取训练数据
#print(csv_data.shape)
#归一化
#print(csv_data.head(5))
csv_norm=csv_data.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
#print(csv_norm.head(5))

#XY分离
Y = csv_norm[['Outcome']]
csv_norm.drop(['Outcome'],axis=1,inplace=True)
X = csv_norm
X = np.array(X)
X = X.tolist()
#array 转换
Y = np.array(Y)
Y = Y.tolist()
#data=np.array(csv_data)
#data=data.tolist()
#解嵌套
Y = list(itertools.chain.from_iterable(Y))



#train_test_split
test_size=0.2
n_train_samples = int(len(X) * (1-test_size))
x_train, x_test = X[:n_train_samples], X[n_train_samples:]
y_train, y_test = Y[:n_train_samples], Y[n_train_samples:]

#10.9这边以上都对了

#矩阵的计算先放一下
# def accuracy(y, y_pred):
    # y = y.reshape(y.shape[0], -1)
    # y_pred = y_pred.reshape(y_pred.shape[0], -1)
    # return np.sum(y == y_pred)/len(y)
#计算欧氏距离
#画图
plt.hist(csv_data['Pregnancies'],  facecolor="blue", edgecolor="black", alpha=0.7)
# 显示图标题
plt.title("Pregnancies")
#plt.show()
plt.savefig('./dialog/Pregnancies.jpg')   #图片的存储
plt.close()

plt.hist(csv_data['Glucose'],  facecolor="blue", edgecolor="black", alpha=0.7)
# 显示图标题
plt.title("Glucose")
#plt.show()
plt.savefig('./dialog/Glucose.jpg')   #图片的存储
plt.close()

plt.hist(csv_data['BloodPressure'],  facecolor="blue", edgecolor="black", alpha=0.7)
# 显示图标题
plt.title("BloodPressure")
#plt.show()
plt.savefig('./dialog/BloodPressure.jpg')   #图片的存储
plt.close()

plt.hist(csv_data['SkinThickness'],  facecolor="blue", edgecolor="black", alpha=0.7)
# 显示图标题
plt.title("SkinThickness")
#plt.show()
plt.savefig('./dialog/SkinThickness.jpg')   #图片的存储
plt.close()


plt.hist(csv_data['Insulin'],  facecolor="blue", edgecolor="black", alpha=0.7)
# 显示图标题
plt.title("Insulin")
#plt.show()
plt.savefig('./dialog/Insulin.jpg')   #图片的存储
plt.close()

plt.hist(csv_data['BMI'],  facecolor="blue", edgecolor="black", alpha=0.7)
# 显示图标题
plt.title("BMI")
#plt.show()
plt.savefig('./dialog/BMI.jpg')   #图片的存储
plt.close()

plt.hist(csv_data['DiabetesPedigreeFunction'],  facecolor="blue", edgecolor="black", alpha=0.7)
# 显示图标题
plt.title("DiabetesPedigreeFunction")
#plt.show()
plt.savefig('./dialog/DiabetesPedigreeFunction.jpg')   #图片的存储
plt.close()

plt.hist(csv_data['Age'],  facecolor="blue", edgecolor="black", alpha=0.7)
# 显示图标题
plt.title("Age")
#plt.show()
plt.savefig('./dialog/Age.jpg')   #图片的存储
plt.close()

plt.hist(csv_data['Outcome'], bins=2, facecolor="blue", edgecolor="black", alpha=0.7)
# 显示图标题
plt.title("Outcome")
#plt.show()
plt.savefig('./dialog/Outcome.jpg')   #图片的存储
plt.close()

# Index(['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
#        'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'],
#       dtype='object')





def distance(a,b):
    result=0.0
    for i in range(len(a)):
        result=result+(a[i]-b[i])**2
    result=result**0.5

    return result

#10.9test
#distance(x_test[0],x_test[1])

#计算一个到整个训练集的距离
def distance_cal(a,X_train):
    distance_set=[]
    for i in range(len(X_train)):
        temp=distance(a,X_train[i])
        distance_set.append(temp)
    return distance_set

#测试
#a = distance_cal(x_test[0],x_train)

#算标签
def get_k_neighbor_labels(distance_set, y_train, k):
    k_neighbor_labels = []
    for i in range(k):
        min_index = distance_set.index(min(distance_set))

        label = y_train[min_index]
        k_neighbor_labels.append(label)
        distance_set[min_index]=max(distance_set)

    return k_neighbor_labels

#print(get_k_neighbor_labels(a,y_train,3))

#找标签
def vote(k_neighbor_labels):
    # print(distances.shape)

    # print(k_neighbor_labels.shape)
    # from collections import Counter
    # lists = ['a', 'a', 'b', 5, 6, 7, 5]
    # a = Counter(lists)
    # print(a)  # Counter({'a': 2, 5: 2, 'b': 1, 6: 1, 7: 1})
    # a.elements()  # 获取a中所有的键,返回的是一个对象,我们可以通过list来转化它
    # a.most_common(2)  # 前两个出现频率最高的元素已经他们的次数,返回的是列表里面嵌套元组
    # a['zz']  # 访问不存在的时候,默认返回0
    # a.update("aa5bzz")  # 更新被统计的对象,即原有的计数值与新增的相加,而不是替换
    # a.subtrct("aaa5z")  # 实现与原有的计数值相减,结果运行为0和负值
    a=Counter(k_neighbor_labels)
    label=a.most_common(1)

    return label[0][0]





#test


#预测
def predict(X_test, X_train, y_train,k):
    y_pred = []
    for sample in X_test:
        distance_set=distance_cal(sample,X_train)
        k_neighbor_labels=get_k_neighbor_labels(distance_set,y_train,k)
        label = vote(k_neighbor_labels)
        y_pred.append(label)
    # print(y_pred)
    return np.array(y_pred)

#10.9计算部分都对了
y_pred=predict(x_test, x_train, y_train,3)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
print("report confusion_matrix:")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
from sklearn.metrics import accuracy_score
print("accuracy:")
print(accuracy_score(y_test, y_pred))
from sklearn.metrics import precision_score
print("precision:")
print(precision_score(y_test, y_pred))


plt.hist(y_pred, bins=2, facecolor="blue", edgecolor="black", alpha=0.7)
# 显示图标题
plt.title("pred_outcome")
#plt.show()
plt.savefig('./dialog/pred_outcome.jpg')   #图片的存储
plt.close()


#runtime
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
end = datetime.datetime.now()
print (end-start)