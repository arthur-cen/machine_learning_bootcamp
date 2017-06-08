from mnist import MNIST
import sklearn.metrics as metrics
import numpy as np

NUM_CLASSES = 10

def load_dataset():
    mndata = MNIST('./data/')
    X_train, labels_train = map(np.array, mndata.load_training())
    X_test, labels_test = map(np.array, mndata.load_testing())
    X_train = X_train/255.0
    X_test = X_test/255.0
    return (X_train, labels_train), (X_test, labels_test)


def train(X_train, y_train, reg=0):
    ''' Build a model from X_train -> y_train '''
    W = np.linalg.inv((X_train.T.dot(X_train) + np.identity(X_train.shape[1]).dot(reg))).dot(X_train.T).dot(y_train)
    return W # dim(W) = d * k ----> 784 * 10 in this case

def one_hot(labels_train):
    '''Convert categorical labels 0,1,2,....9 to standard basis vectors in R^{10} '''
    '''Check out later'''
    return np.array([np.eye(NUM_CLASSES)[ind] for ind in labels_train])

def predict(model, X):
    ''' From model and data points, output prediction vectors '''
    pred_labels_oh = X.dot(model)
    pred_labels = np.apply_along_axis(np.argmax, axis = 1, arr = pred_labels_oh)
    return pred_labels
if __name__ == "__main__":
    (X_train, labels_train), (X_test, labels_test) = load_dataset()
    y_train = one_hot(labels_train)
    model = train(X_train, y_train, reg = 0.9)
    y_test = one_hot(labels_test)

    pred_labels_train = predict(model, X_train)
    pred_labels_test = predict(model, X_test)


    print("Train accuracy: {0}".format(metrics.accuracy_score(labels_train, pred_labels_train)))
    print("Test accuracy: {0}".format(metrics.accuracy_score(labels_test, pred_labels_test)))
