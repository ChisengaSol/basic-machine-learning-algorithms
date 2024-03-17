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

         

def run_logistic_regression(x_train, y_train,leaning_rate,n_epochs):
    model1 = LogisticRegression(lr=leaning_rate ,n_epochs=n_epochs)
    model1.batch_grad(x_train, y_train)

    y_pred_train = model1.predict(x_train)
    train_acc = model1.accuracy(y_train, y_pred_train)
    print(f"The training accuracy when we use batch gradient without momentum is: {train_acc}%")

    model2 = LogisticRegression(lr=leaning_rate ,n_epochs=n_epochs)
    model2.batch_grad_with_momentum(x_train, y_train)

    y_pred_train1 = model2.predict(x_train)
    train_acc1 = model2.accuracy(y_train, y_pred_train1)
    print(f"The training accuracy when we use batch gradient with momentum is: {train_acc1}%")

    model3 = LogisticRegression(lr=leaning_rate ,n_epochs=n_epochs)
    model3.train_sgd(x_train, y_train)

    y_pred_train2 = model3.predict(x_train)
    train_acc2 = model3.accuracy(y_train, y_pred_train2)
    print(f"The training accuracy when we use sgd is: {train_acc2:.2f}%")

    model4 = LogisticRegression(lr=leaning_rate ,n_epochs=n_epochs)
    model4.train_sgd_with_momentum(x_train, y_train)

    y_pred_train4 = model3.predict(x_train)
    train_acc4 = model3.accuracy(y_train, y_pred_train4)
    print(f"The training accuracy when we use sgd with momentum is: {train_acc4:.2f}%")