import pandas as pd
import matplotlib.pyplot as plt
import sklearn

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.datasets import load_breast_cancer

# x = [i for i in range(10)]
# print("Values of X:",x)
# y = [2 ** i for i in range(10)]
# print("Values of Y:",y)
#
# plt.title('Plot Chart Show Window')
# plt.xlabel('X-Axis')
# plt.ylabel('Y-Axis')
# plt.plot(x,y)
# plt.show()
#
# plt.title('Scatter Chart Show Window')
# plt.scatter(x,y)
# plt.show()

data = load_breast_cancer()
list(data.target_names)

clf = DecisionTreeClassifier(max_depth=3) # max_depth는 트리의 최대 레벨 수입니다.
clf.fit(data.data, data.target)

plt.figure(figsize=(25,10))
a = plot_tree(clf,
              feature_names=data.feature_names,
              class_names=data.target_names,
              filled=True,
              fontsize=12)
plt.show()


#데이터 셋을 훈련 셋이랑 검증 셋 나누기

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

X = data.data
y = data.target

print(X.shape)
print(y.shape,'\n')

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

print(X_train.shape)
print(X_test.shape,'\n')

print(y_train.shape)
print(y_test.shape)


