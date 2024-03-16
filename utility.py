import numpy as np
from logistic_regression import LogisticRegression 
from linear_regression import LinearRegression

def generate_data(n=1000):
    np.random.seed(0)
    x = np.linspace(-5.0, 5.0, n).reshape(-1, 1)
    y = (29 * x + 30 * np.random.rand(n, 1)).squeeze()
    x = np.hstack((np.ones_like(x), x))
    return x, y

def split_data(x, y, train_perc=0.8):
    N = x.shape[0]
    train_size = round(train_perc * N)
    x_train, y_train = x[:train_size, :], y[:train_size]
    x_test, y_test = x[train_size:, :], y[train_size:]
    return x_train, y_train, x_test, y_test

def split_data_1(x, y, train_perc=0.8):
    N = x.shape[0]
    idx = np.random.permutation(x.shape[0])
    x, y = x[idx], y[idx]
    train_size = round(train_perc * N)
    x_train, y_train = x[:train_size, :], y[:train_size]
    x_test, y_test = x[train_size:, :], y[train_size:]
    return x_train, y_train, x_test, y_test

def run_linear_regression(x_train, y_train,leaning_rate,n_epochs):
    m = LinearRegression(lr = leaning_rate,n_epochs = n_epochs)
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

def run_logistic_regression(x_train, y_train,leaning_rate,n_epochs):
    model = LogisticRegression(lr=leaning_rate ,n_epochs=n_epochs)
    model.fit(x_train, y_train)

    y_pred_train = model.predict(x_train)
    train_acc = model.accuracy(y_train, y_pred_train)
    return f"The training accuracy is: {train_acc}%"