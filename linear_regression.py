import numpy as np

def compute_loss(y_true, y_pred):
    """
    MSE : sum((y_true-y_pred)^2/N)
    """
    mse = np.mean(np.square(y_true-y_pred))
    return mse

def generate_metrics(y_true, y_pred):
    """
    Calculates the R2 Score 
    Input:
    =====
        1. y_true
        2. y_pred
    Output:
    ======
        1. R2 Score
    """
    sse = np.sum(np.square(y_true-y_pred))
    sst = np.sum(np.square(y_true-np.mean(y_true)))
    r2_score = 1-(sse/sst)
    return r2_score

def generate_dataset():
    """
    Generates a dataset for Regression
    Output:
    =====
        1. X
        2. y
    """

    from sklearn.datasets import make_regression

    X, y = make_regression(n_samples=1000,
                           n_features=3,
                           n_informative=2,
                           n_targets=1,
                           random_state=9)
    y = y.reshape(-1, 1)
    return X, y


def generate_train_batch(X, y, batch_size):
    """
    Generates batches of training data
    Input:
    =====
        1. X
        2. y
        3. batch_size: Mini batch size
    Output:
    =====
        1. X_batch
        2. y_batch
    """
    import random

    indices = np.arange(X.shape[0]).tolist()
    random.shuffle(indices)
    batch_count = X.shape[0]//batch_size

    X, y = X[indices], y[indices]
    for i in range(0, batch_count):
        if i != batch_count-1:
            X_batch, y_batch = X[i*batch_size:(i+1)*batch_size], y[i*batch_size:(i+1)*batch_size]
            yield X_batch, y_batch
        else:
            X_batch, y_batch = X[-batch_size:], y[-batch_size:]
            yield X_batch, y_batch

def compute_gradient(X, y, y_pred):
    """
    Computes Gradient of Loss function : MSE
    Input:
    =====
        1. X 
        2. y
        3. y_pred
    Output:
    ======
        1. gradient (Shape: (X.shape[1], 1))
    """
    gradient = 2*np.dot(X.T, (y_pred-y).reshape(-1, 1))/X.shape[0]
    return gradient

def initialize_weights(X):
    """
    Initializes weights to be trained.
    Input:
    =====
        1. X
    Output:
    ======
        1. w : Randomly initialized weight matrix. (Shape : (X.shape[1],1))
    """
    w = np.random.rand(X.shape[1], 1)
    return w

class LinearRegression:
    """
    Blue Print for Linear Regression 
    States:
        1. coeff : Coefficients learnt by the model after training.
    """
    def __init__(self, alpha, batch_size, iterations, coeff):
        self.alpha = alpha
        self.batch_size = batch_size
        self.iterations = iterations
        self.coeff = coeff

    def train_model(self, X=None, y=None):
        """
        Method for model training.
        Input:
        =====
            1. X
            2. y
        """
        from sklearn.model_selection import train_test_split
        train_X, test_val_X, train_y, test_val_y = train_test_split(X, y,
                                                                    random_state=9,
                                                                    test_size=0.2)
        val_X, test_X, val_y, test_y = train_test_split(test_val_X, test_val_y,
                                                        random_state=9,
                                                        test_size=0.5)                                              
        val_loss_dict, train_loss_dict = dict(), dict()
        for i in range(self.iterations):
            for X_batch, y_batch in generate_train_batch(train_X, train_y, self.batch_size):
                y_batch_pred = np.dot(X_batch, self.coeff)
                gradient = compute_gradient(X_batch, y_batch, y_batch_pred)
                self.coeff = self.coeff-self.alpha*gradient
            # Get predictions of validation.
            val_y_pred = np.dot(val_X, self.coeff)
            val_loss = compute_loss(val_y, val_y_pred)
            val_loss_dict[i] = val_loss
            # Get predictions of train.
            train_y_pred = np.dot(train_X, self.coeff)
            train_loss = compute_loss(train_y, train_y_pred)
            train_loss_dict[i] = train_loss
            # Get r2 score of train.
            r2_train = generate_metrics(train_y, train_y_pred)
            # Get r2 score of validation.
            r2_val = generate_metrics(val_y, val_y_pred)
            if i%(self.iterations//10) == 0:
                print("""Iteration : {} ... Train Loss : {} ...
                      Val Loss : {} ... Train R2 : {} ... Val R2 : {}""".\
                    format(i, train_loss, val_loss, r2_train, r2_val))
            # Early Stopping.
            if i > 2:
                if (val_loss_dict[i-1]-val_loss_dict[i]) < 1e-8:
                    print("Early Stopping, found no improvement")
                    break
        # Get Final metrics of test.
        test_y_pred = np.dot(test_X, self.coeff)
        test_loss = compute_loss(test_y, test_y_pred)
        r2_test = generate_metrics(test_y, test_y_pred)
        print("Iteration : {} ... Test Loss : {} ... Test R2 : {}".format(i, test_loss, r2_test))

if __name__ == "__main__":
    features, target = generate_dataset()
    initial_w = initialize_weights(features)
    lr = LinearRegression(alpha=1e-3, iterations=100, batch_size=32, coeff=initial_w)
    lr.train_model(features, target)
    print("Coeff : {}".format(lr.coeff))