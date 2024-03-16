import numpy as np
class LinearRegression:
    def __init__(self, lr, n_epochs):
        self.lr = lr
        self.n_epochs = n_epochs
        self.weight = None

    def linear_function(self, x, theta):

        return x @ theta

    def initialize_theta(self, D):
        return np.zeros((D, 1))

    def mean_squared_error(self, y_true, y_pred):
        return np.mean(np.square(y_true - y_pred))

    def get_momentum(self, momentum, beta, grad):
        return momentum * beta + (1 - beta) * grad

    def per_sample_gradient(self, xi, yi, theta):
        return 2 * xi.T @ (self.linear_function(xi, theta) - yi)

    def shuffle_data(self, x, y):
        N, _ = x.shape
        shuffled_idx = np.random.permutation(N)
        return x[shuffled_idx], y[shuffled_idx]

    def update_function(self, theta, grads, step_size):
        return theta - step_size * grads

    def batch_gradient(self, x, y, theta):
        return (2 / y.shape[0]) * x.T @ (self.linear_function(x, theta) - y)

    def train_batch_gradient(self, x, y, step_size=0.1, beta=0.99, with_momentum=False):
        N, D = x.shape
        theta = self.initialize_theta(D)
        losses = []
        for epoch in range(self.n_epochs):
            momentum = 0.0
            ypred = self.linear_function(x, theta)
            loss = self.mean_squared_error(y, ypred)
            grad = self.batch_gradient(x, y, theta)
            if with_momentum:
                momentum = self.get_momentum(momentum, beta, grad)
                theta = self.update_function(theta, momentum, step_size)
            else:
                theta = self.update_function(theta, grad, step_size)

            losses.append(loss)
            print(f"\nEpoch {epoch}, loss {loss}")
        return losses

    def train_sgd(self, x, y, step_size=0.1, beta=0.99, with_momentum=False):
        N, D = x.shape
        theta = self.initialize_theta(D)
        losses = []
        epoch = 0
        loss_tolerance = 0.001
        avg_loss = float("inf")

        while epoch < self.n_epochs and avg_loss > loss_tolerance:
            running_loss = 0.0
            shuffled_x, shuffled_y = self.shuffle_data(x, y)
            momentum = 0.0

            for idx in range(shuffled_x.shape[0]):
                sample_x = shuffled_x[idx].reshape(-1, D)
                sample_y = shuffled_y[idx].reshape(-1, 1)
                ypred = self.linear_function(sample_x, theta)
                loss = self.mean_squared_error(sample_y, ypred)
                running_loss += loss
                grad = self.per_sample_gradient(sample_x, sample_y, theta)
                if with_momentum:
                    momentum = self.get_momentum(momentum, beta, grad)
                    theta = self.update_function(theta, momentum, step_size)
                else:
                    theta = self.update_function(theta, grad, step_size)

            avg_loss = running_loss / x.shape[0]
            losses.append(avg_loss)
            print(f"Epoch {epoch}, loss {avg_loss}")

            epoch += 1

        return losses

    def minibatch_gradient_descent(
        self, x, y, step_size=0.1, batch_size=3, beta=0.99, with_momentum=False
    ):
        N, D = x.shape
        theta = self.initialize_theta(D)
        losses = []
        x, y = self.shuffle_data(x, y)

        for epoch in range(self.n_epochs):
            running_loss = 0.0
            momentum = 0.0

            for batch_idx in range(0, N, batch_size):
                x_batch = x[batch_idx : batch_idx + batch_size].reshape(-1, D)
                y_batch = y[batch_idx : batch_idx + batch_size].reshape(-1, 1)

                ypred = self.linear_function(x_batch, theta)
                loss = self.mean_squared_error(y_batch, ypred)
                grad = self.batch_gradient(x_batch, y_batch, theta)
                if with_momentum:
                    momentum = self.get_momentum(momentum, beta, grad)
                    theta = self.update_function(theta, momentum, step_size)
                else:
                    theta = self.update_function(theta, grad, step_size)
                running_loss += loss * x_batch.shape[0]

            avg_loss = running_loss / N
            losses.append(avg_loss)
            print(f"\nEpoch {epoch}, loss {avg_loss}")

        return losses
