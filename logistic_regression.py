import numpy as np
class LogisticRegression:
    def __init__(self, lr, n_epochs):
        self.lr = lr
        self.n_epochs = n_epochs
        self.w = None
        self.train_losses = []

    def add_ones(self, x):
        return np.hstack((np.ones((x.shape[0], 1)), x))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x @ self.w))

    def cross_entropy(self, x, y_true):
        y_pred = self.sigmoid(x)
        epsilon = 1e-15
        loss = -np.mean(
            y_true * np.log(y_pred + epsilon)
            + (1 - y_true) * np.log(1 - y_pred + epsilon)
        )
        return loss

    def predict_proba(self, x):
        x = self.add_ones(x)
        return self.sigmoid(x)

    def predict(self, x):
        return (self.predict_proba(x) >= 0.5).astype(int)
    
    def get_momentum(self, momentum, beta, grad):
        return momentum * beta + (1 - beta) * grad
    
    def shuffle_data(self, x, y):
        idx = np.random.permutation(len(x))
        return x[idx], y[idx]
    
    def batch_gradient_descent(self, x, y, with_momentum=False, beta=0.99):
        x = self.add_ones(x)
        self.w = np.zeros((x.shape[1], 1))
        for epoch in range(self.n_epochs):
            ypred = self.sigmoid(x)
            grad = (-1 / x.shape[0]) * (x.T @ (y - ypred))
            if with_momentum:
                momentum = 0.0
                momentum = self.get_momentum(momentum, beta, grad)
                self.w = self.w - self.lr * momentum
            else:
                self.w = self.w - self.lr * grad
            loss = self.cross_entropy(x, y)
            self.train_losses.append(loss)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, loss {loss}")

    def stochastic_gradient_descent(self, x, y, with_momentum=False, beta=0.99):
        x = self.add_ones(x)
        self.w = np.zeros((x.shape[1], 1))
        for epoch in range(self.n_epochs):
            running_loss = 0.0
            shuffled_x, shuffled_y = self.shuffle_data(x, y)
            for idx in range(shuffled_x.shape[0]):
                sample_x = shuffled_x[idx].reshape(-1, x.shape[1])
                sample_y = shuffled_y[idx].reshape(-1, 1)
                ypred = self.sigmoid(sample_x)
                grad = (-1) * (sample_x.T @ (sample_y - ypred))
                if with_momentum:
                    momentum = 0.0
                    momentum = self.get_momentum(momentum, beta, grad)
                    self.w = self.w - self.lr * momentum
                else:
                    self.w = self.w - self.lr * grad
                loss = self.cross_entropy(sample_x, sample_y)
                running_loss += loss
            avg_loss = running_loss / x.shape[0]
            self.train_losses.append(avg_loss)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, loss {avg_loss}")

    def train(self, x, y, method='batch', with_momentum=False, beta=0.99):
        if method == 'batch':
            self.batch_gradient_descent(x, y, with_momentum, beta)
        elif method == 'sgd':
            self.stochastic_gradient_descent(x, y, with_momentum, beta)

    def accuracy(self, y_true, y_pred):
        acc = np.mean(y_true == y_pred) * 100
        return acc