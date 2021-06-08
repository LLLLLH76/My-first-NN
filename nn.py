import numpy as np
seed = 1
np.random.seed(seed)
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def sigmoid(x):
    y = np.array(x)
    return (1+np.e**(-y))**(-1)


def deriv_sigmoid(x):
    y = np.array(x)
    return np.e**(-y) * (1 + np.e**(-y))**(-2)

def mse_loss(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return (((y_true - y_pred)**2).sum()) / len(y_true)
    

def accuracy(y_true, y_pred):
    return sum(y_true == y_pred) / len(y_true)

def to_onehot(y):
    y = y.reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(y)
    y = enc.transform(y).toarray()
    return y

class NeuralNetwork():
    def __init__(self, d, q, l):
        # weights
        self.v = np.random.randn(d, q)
        self.w = np.random.randn(q, l)
        # biases
        self.gamma = np.random.randn(q)
        self.theta = np.random.randn(l)

    def predict(self, X):
        '''
        X: shape (n_samples, d)
        returns: shape (n_samples, l)
        '''
        hidden_layer_input_matrix = X @ self.v
        hidden_layer_output_matrix = sigmoid(hidden_layer_input_matrix - self.gamma)
        output_layer_input_matrix = hidden_layer_output_matrix @ self.w
        output_layer_output_matrix = sigmoid(output_layer_input_matrix - self.theta)
        return output_layer_output_matrix
        
    def train(self, X, y, learning_rate = 1, epochs = 500):
        '''
        X: shape (n_samples, d)
        y: shape (n_samples, l)
        '''
        for epoch in range(epochs):
            # output layer gradient
            alpha = X @ self.v
            b = sigmoid(alpha - self.gamma)
            beta = b @ self.w
            y_preds = sigmoid(beta - self.theta)
            g = y_preds * (1 - y_preds) * (y - y_preds)
            # hidden layer gradient
            e = b * ( 1 - b ) * ( g @ self.w.T)
            # update weights and biases
            self.w += learning_rate * len(X)**(-1) * b.T @ g
            self.v += learning_rate * len(X)**(-1) * X.T @ e    
            self.gamma -= learning_rate * e.mean(axis = 0)
            self.theta -= learning_rate * g.mean(axis = 0)
            # calcalate loss
            if epoch % 10 == 0:
                y_preds = self.predict(X)
                loss = mse_loss(y, y_preds)
                print("Epoch %d loss: %.3f"%(epoch, loss))
    
if __name__ ==  '__main__':
    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)
    y_train = to_onehot(y_train)

    n_features = X.shape[1]
    n_hidden_layer_size = 100
    n_outputs = len(np.unique(y)) # unique 去重  n_outputs = 10
    network = NeuralNetwork(d = n_features, q = n_hidden_layer_size, l = n_outputs)
    network.train(X_train, y_train, learning_rate = 0.8, epochs = 500)

    y_pred = network.predict(X_test)
    y_pred_class = np.argmax(y_pred, axis=1)
    mse = mse_loss(to_onehot(y_test), y_pred)
    print("\nTesting MSE: {:.3f}".format(mse))
    acc = accuracy(y_test, y_pred_class) * 100
    print("\nTesting Accuracy: {:.3f} %".format(acc))
    