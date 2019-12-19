import numpy as np

def sigmoid(x):
    """
    Applies Sigmoid Transformation
    A = 1/(1+e^(-x))
    """
    return 1/(1+np.exp(-x))

def compute_loss(y_true, y_pred):
    """
    Computes the log-loss.
    J(w) = -(ylog(p)+(1-y)log(1-p))
    Input:
    =====
        1. y_true
        2. y_pred
    
    Output:
    ======
        1. Computed Loss
    """
    loss = -np.sum(y_true*np.log(y_pred)+(1-y_true)*np.log(1-y_pred))/(y_true.shape[0])
    return loss

def compute_gradient(w, y_true, X):
    """
    Performs Numerical Differentiation
    Input:
    ======
        1. w : Current Weight Matrix
        2. y_true
        3. X
    Output:
    =======
        1. Gradient
    """
    epsilon = 1e-3
    y_pred = sigmoid(np.dot(X, w))
    y_pred_h = sigmoid(np.dot(X, w+epsilon))
    gradient = (compute_loss(y_true, y_pred_h) - compute_loss(y_true, y_pred))/epsilon
    return gradient

def generate_metrics(y_true, y_pred, save_fig=False):
    """
    Calculates the F1 Score 
    Input:
    =====
        1. y_true
        2. y_pred
    Output:
    ======
        1. F1 Score
    """
    from sklearn.metrics import f1_score
    y_pred_class = np.where(y_pred>0.5, 1, 0)
    return f1_score(y_true, y_pred_class)

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

def save_plots(train_loss, validation_loss, train_label=None, validation_label=None):
        # Save Plots
        import matplotlib.pyplot as plt
        import os

        plt.plot(train_loss.keys(), train_loss.values(), label = "Train Loss; F1 :"+str(train_label))
        plt.plot(validation_loss.keys(), validation_loss.values(), label = "Validation Loss; F1 :"+str(validation_label))
        plt.xlabel("Epoch")
        plt.ylabel("Binary Cross Entropy")
        plt.title("Change in Loss after each epoch")
        plt.legend()
        if not os.path.exists("results/"):
            os.mkdir("results/")
        plt.savefig("results/LogisticRegression_Loss.png")
        plt.close()

def generate_dataset():
    """
    Generates a dataset for Regression
    Output:
    =====
        1. X
        2. y
    """

    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=1000, 
                               n_features=3,
                               n_informative=2, 
                               n_redundant=0,
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

def standardize(X1, X2):
    X1 = (X1-np.mean(X2))/np.std(X2)
    return X1

class LogisticRegression:
    """
    Blue Print for Logistic Regression 
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
                                                                    test_size=0.2,
                                                                    stratify=y)

        val_X, test_X, val_y, test_y = train_test_split(test_val_X, test_val_y,
                                                        random_state=5,
                                                        test_size=0.5,
                                                        stratify=test_val_y
                                                        )
        train_X = standardize(train_X, train_X)
        test_X = standardize(test_X, train_X)
        val_X = standardize(val_X, train_X)
 

        # Start Training.                                  
        val_loss_dict, train_loss_dict = dict(), dict()
        for i in range(1, self.iterations+1):
            for X_batch, y_batch in generate_train_batch(train_X, train_y, self.batch_size):
                gradient = compute_gradient(self.coeff, y_batch, X_batch)
                self.coeff = self.coeff-self.alpha*gradient
            # Get predictions of validation.
            val_y_pred = sigmoid(np.dot(val_X, self.coeff))
            val_loss = compute_loss(val_y, val_y_pred)
            val_loss_dict[i] = val_loss
            # Get predictions of train.
            train_y_pred = sigmoid(np.dot(train_X, self.coeff))
            train_loss = compute_loss(train_y, train_y_pred)
            train_loss_dict[i] = train_loss
            # Get f1 score of train.
            f1_train = generate_metrics(train_y, train_y_pred)
            # Get f1 score of validation.
            f1_val = generate_metrics(val_y, val_y_pred)
            if i%(self.iterations//10) == 0:
                print("""Iteration : {} ... Train Loss : {} ... \
                      Val Loss : {} ... Train F1 : {} ... Val F1 : {}""".\
                    format(i, train_loss, val_loss, f1_train, f1_val))
            # Early Stopping.
            if i > 2 and i <self.iterations+1:
                if (val_loss_dict[i-1]-val_loss_dict[i]) < 1e-8:
                    print("Early Stopping, found no improvement")
                    break
        # Get Final metrics of test.
        test_y_pred = sigmoid(np.dot(test_X, self.coeff))
        test_loss = compute_loss(test_y, test_y_pred)
        f1_test = generate_metrics(test_y, test_y_pred)
        print("""Iteration : {} ... Train Loss : {} ... \
                 Val Loss : {} ... Train F1 : {} ... Val F1 : {}""".\
                 format(i, train_loss, val_loss, f1_train, f1_val))
        print("Iteration : {} ... Test Loss : {} ... Test F1 : {}".format(i, test_loss, f1_test))
        save_plots(train_loss_dict, val_loss_dict, f1_train, f1_val)

if __name__ == "__main__":
    X, y = generate_dataset()
    w = initialize_weights(X)
    lr = LogisticRegression(alpha=1e-3, batch_size=32, iterations=1000, coeff = w)
    lr.train_model(X, y)
    print(lr.coeff)