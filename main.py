import linear_regression, logistic_regression
import numpy as np
import matplotlib.pyplot as plt
from utility import generate_data, split_data, split_data_1

np.random.seed(0)

x, y = generate_data()
# check the shape
print((x.shape, y.shape))

x_train, y_train, x_test, y_test = split_data(x, y)
print(
    f"x_train:{x_train.shape}, y_train: {y_train.shape}, x_test: {x_test.shape}, y_test: {y_test.shape}"
)

plt.plot(x_train[:, 1], y_train)
plt.show()

m = linear_regression.Linear_regression(0.01, 10)
batch_without_momentum = m.train_batch_gradient(x_train, y_train)
m.plot_loss(batch_without_momentum, "Batch gradient without momentum")

batch_with_momentum = m.train_batch_gradient(x_train, y_train, with_momentum=True)
m.plot_loss(batch_with_momentum, "Batch gradient with momentum")

sgd_without_momentum = m.train_sgd(x_train, y_train)
m.plot_loss(sgd_without_momentum, "sdg without momentum")

sgd_with_momentum = m.train_sgd(x_train, y_train, with_momentum=True)
m.plot_loss(sgd_with_momentum, "sdg with momentum")

mini_batch_without_momentum = m.minibatch_gradient_descent(x_train, y_train)
m.plot_loss(mini_batch_without_momentum, "mini batch without momentum")

mini_batch_with_momentum = m.minibatch_gradient_descent(
    x_train, y_train, with_momentum=True
)
m.plot_loss(mini_batch_with_momentum, "mini batch with momentum")


from sklearn.datasets import make_classification

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

model = logistic_regression.LogisticRegression(lr=0.1, n_epochs=1000)
model.fit(x_train, y_train)

# Evaluate the model on the test set
y_pred_train = model.predict(x_train)
train_acc = model.accuracy(y_train, y_pred_train)
print(f"The training accuracy is: {train_acc}%")
