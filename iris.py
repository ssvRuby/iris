import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import mglearn

plt.ion()
iris_dataset = load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

print('Форма массива X_train: {}'.format(X_train.shape))
print('Форма массива y_train: {}'.format(y_train.shape))
print('Форма массива X_test: {}'.format(X_test.shape))
print('Форма массива y_test: {}'.format(y_test.shape))

iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)

grr = pd.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o', hist_kwds={'bins': 20}, s=60, alpha=.8,
                        cmap=mglearn.cm3)

# print("Key from Iris datasets: \n{}".format(iris_dataset.keys()))
# print(iris_dataset['DESCR'][:193])
# print('===========================')
# print(iris_dataset['target_names'])
# print('===========================')
# print(iris_dataset['feature_names'])
# print('===========================')
# print(iris_dataset['target'])
# print('===========================')
print(iris_dataset['data'][:5])

