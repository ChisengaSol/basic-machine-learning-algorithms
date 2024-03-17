import numpy as np
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
    
    def get_momentum(self, momentum, beta, grad):
        return momentum * beta + (1 - beta) * grad
    
    def shuffle_data(self, x, y):
        N, _ = x.shape
        shuffled_idx = np.random.permutation(N)
        return x[shuffled_idx], y[shuffled_idx]
    
    #handle gradient desent
    def gradient_descent(self, x, y,with_momentum = False,beta=0.99):
        ypred = self.sigmoid(x)
        grad = (-1 / x.shape[0]) * (x.T @ (y - ypred))
        if with_momentum:
            momentum = 0.0
            momentum = self.get_momentum(momentum, beta, grad)
            self.w = self.w - self.lr * momentum
        else:
            self.w = self.w - self.lr * grad

    #function fits the model
    def batch_grad(self, x, y):
        # Add ones to x
        x = self.add_ones(x)

        # Initialize w to zeros vector
        self.w = np.zeros((x.shape[1], 1))

        for epoch in range(self.n_epochs):
            self.gradient_descent(x, y)
            loss = self.cross_entropy(x, y)
            self.train_losses.append(loss)

            if epoch % 100 == 0:
                print(f"loss for epoch {epoch}: {loss}")

    def batch_grad_with_momentum(self, x, y):
        # Add ones to x
        x = self.add_ones(x)

        # Initialize w to zeros vector
        self.w = np.zeros((x.shape[1], 1))

        for epoch in range(self.n_epochs):
            self.gradient_descent(x, y,with_momentum=True)
            loss = self.cross_entropy(x, y)
            self.train_losses.append(loss)

            if epoch % 100 == 0:
                print(f"loss for epoch {epoch}: {loss}")

    def train_sgd(self, x, y):
        # Add ones to x
        x = self.add_ones(x)

        # Initialize w to zeros vector
        self.w = np.zeros((x.shape[1], 1))
        epoch = 0
        loss_tolerance = 0.001
        avg_loss = float("inf")

        losses = []

        while epoch < self.n_epochs and avg_loss > loss_tolerance:
            running_loss = 0.0
            shuffled_x, shuffled_y = self.shuffle_data(x, y)
            momentum = 0.0
            beta = 0.99

            for idx in range(shuffled_x.shape[0]):
                sample_x = shuffled_x[idx].reshape(-1, x.shape[1])
                sample_y = shuffled_y[idx].reshape(-1, 1)
                ypred = self.sigmoid(sample_x)
                grad = (-1) * (sample_x.T @ (sample_y - ypred))
                momentum = self.get_momentum(momentum, beta, grad)
                loss = self.cross_entropy(sample_x, sample_x)
                running_loss += loss
                self.w = self.w - self.lr * momentum

            avg_loss = running_loss / x.shape[0]
            losses.append(avg_loss)
            print(f"Epoch {epoch}, loss {avg_loss}")

            epoch += 1

    def train_sgd_with_momentum(self, x, y):
        # Add ones to x
        x = self.add_ones(x)

        # Initialize w to zeros vector
        self.w = np.zeros((x.shape[1], 1))
        epoch = 0
        loss_tolerance = 0.001
        avg_loss = float("inf")

        losses = []

        while epoch < self.n_epochs and avg_loss > loss_tolerance:
            running_loss = 0.0
            shuffled_x, shuffled_y = self.shuffle_data(x, y)

            for idx in range(shuffled_x.shape[0]):
                sample_x = shuffled_x[idx].reshape(-1, x.shape[1])
                sample_y = shuffled_y[idx].reshape(-1, 1)
                ypred = self.sigmoid(sample_x)
                grad = (-1) * (sample_x.T @ (sample_y - ypred))
                loss = self.cross_entropy(sample_x, sample_x)
                running_loss += loss
                self.w = self.w - self.lr * grad

            avg_loss = running_loss / x.shape[0]
            losses.append(avg_loss)
            print(f"Epoch {epoch}, loss {avg_loss}")

            epoch += 1


    def accuracy(self, y_true, y_pred):
        acc = np.mean(y_true == y_pred) * 100
        return acc
