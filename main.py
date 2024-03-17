import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from utility import generate_data, split_data, split_data_1, run_logistic_regression, run_linear_regression

np.random.seed(0)

x, y = generate_data()
# check the shape
print((x.shape, y.shape))

x_train, y_train, x_test, y_test = split_data(x, y)
print(
    f"x_train:{x_train.shape}, y_train: {y_train.shape}, x_test: {x_test.shape}, y_test: {y_test.shape}"
)

# plt.plot(x_train[:, 1], y_train)
# plt.show()

X_class, y_class = make_classification(
    n_samples=100,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    random_state=1,
    n_clusters_per_class=1,
)

x_train, y_train, x_test, y_test = split_data_1(X_class, y_class)
print(
    f"x_train:{x_train.shape}, y_train: {y_train.shape}, x_test: {x_test.shape}, y_test: {y_test.shape}"
)

#linear regression
run_linear_regression(x_train = x_train, y_train = y_train, leaning_rate = 0.01,n_epochs = 10)

#logistic regression
run_logistic_regression(x_train = x_train, y_train = y_train,learning_rate=0.01, n_epochs=1000)




