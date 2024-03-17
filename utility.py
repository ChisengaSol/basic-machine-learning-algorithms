import numpy as np
from logistic_regression import LogisticRegression 
from linear_regression import LinearRegression
import matplotlib.pyplot as plt

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

    batch_with_momentum = m.train_batch_gradient(x_train, y_train, with_momentum=True)

    sgd_without_momentum = m.train_sgd(x_train, y_train)

    sgd_with_momentum = m.train_sgd(x_train, y_train, with_momentum=True)

    mini_batch_without_momentum = m.minibatch_gradient_descent(x_train, y_train)

    mini_batch_with_momentum = m.minibatch_gradient_descent(
        x_train, y_train, with_momentum=True
    )

    #gradient descents to be plotted on one figure
    #key is subplot index, list cotains type of gradient and its title
    losses_plots = {1:[batch_without_momentum,"Batch gradient without momentum"],
                    2:[batch_with_momentum,"Batch gradient with momentum"],
                    3:[sgd_without_momentum,"Sdg without momentum"],
                    4:[sgd_with_momentum,"Sgd with momentum"],
                    5:[mini_batch_without_momentum,"Mini batch without momentum"],
                    6:[mini_batch_with_momentum,"mini batch with momentum"],
                    }
     # Adjust size of the matplotlib figure window
    fig = plt.figure(figsize=(10, 10)) 

    for plt_idx, value in losses_plots.items():
        fig.add_subplot(3, 2, plt_idx)
        plt.plot(value[0])
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title(value[1])

    
    
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.4, hspace=0.5)
    plt.suptitle("Linear regression losses using different types of gradient descent")
    plt.show()

         

def run_logistic_regression(x_train, y_train, learning_rate, n_epochs):
    model = LogisticRegression(lr=learning_rate, n_epochs=n_epochs)

    print("Training using Batch Gradient Descent:")
    model.train(x_train, y_train, method='batch')
    train_acc = model.accuracy(y_train, model.predict(x_train))
    print(f"Training accuracy: {train_acc:.2f}%")

    print("Training using Batch Gradient Descent with Momentum:")
    model.train(x_train, y_train, method='batch', with_momentum=True)
    train_acc = model.accuracy(y_train, model.predict(x_train))
    print(f"Training accuracy: {train_acc:.2f}%")

    print("Training using Stochastic Gradient Descent:")
    model.train(x_train, y_train, method='sgd')
    train_acc = model.accuracy(y_train, model.predict(x_train))
    print(f"Training accuracy: {train_acc:.2f}%")

    print("Training using Stochastic Gradient Descent with Momentum:")
    model.train(x_train, y_train, method='sgd', with_momentum=True)
    train_acc = model.accuracy(y_train, model.predict(x_train))
    print(f"Training accuracy: {train_acc:.2f}%")
