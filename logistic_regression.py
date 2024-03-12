import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)


class LogisticRegression:
    def __init__(self, lr, n_epochs):
        self.lr = lr
        self.n_epochs = n_epochs
        self.train_losses = []
        self.w = None

    def add_ones(self, x):
        return np.hstack((np.ones((x.shape[0], 1)), x))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x @ self.w))

    def cross_entropy(self, x, y_true):
        y_pred = self.sigmoid(x)
        # epsilon value to prevent division by zero
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

    def fit(self, x, y):
        # Add ones to x
        x = self.add_ones(x)

        # Initialize w to zeros vector
        self.w = np.zeros((x.shape[1], 1))

        for epoch in range(self.n_epochs):
            ypred = self.sigmoid(x)
            grad = (-1 / x.shape[0]) * (x.T @ (y - ypred))
            self.w = self.w - self.lr * grad
            loss = self.cross_entropy(x, y)
            self.train_losses.append(loss)

            if epoch % 100 == 0:
                print(f"loss for epoch {epoch}: {loss}")

    def accuracy(self, y_true, y_pred):
        acc = np.mean(y_true == y_pred) * 100
        return acc
