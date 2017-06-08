from mnist import MNIST
import sklearn.metrics as metrics
import numpy as np
import matplotlib.pyplot as plt
import scipy

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
    W = np.linalg.inv((X_train.T.dot(X_train) + reg * np.eye(X_train.shape[1]))).dot(X_train.T).dot(y_train)
    return W
def train_gd(X_train, y_train, alpha=0.1, reg=0, num_iter=10000, D=5000):
    ''' Build a model from X_train -> y_train using batch gradient descent '''
    # X --  N * D
    # y --- N * k
    # W arbiturally choose one D * k
    W = np.ones([D, NUM_CLASSES])
    a = X_train.T.dot(X_train) + reg * np.eye(D)
    b = X_train.T.dot(y_train)
    for i in range(0, num_iter):
        gradient = a.dot(W) - b
        W =  W - alpha * gradient
    return W

def train_gd_plot(X_train, y_train, alpha=0.1, reg=0, num_iter=10000, D=5000):
    ''' Build a model from X_train -> y_train using batch gradient descent '''
    # X --  N * D
    # y --- N * k
    # W arbiturally choose one D * k
    global err_gd_list
    N = X_train.shape[0]
    W = np.ones([D, NUM_CLASSES])
    a = X_train.T.dot(X_train) + reg * np.eye(D)
    b = X_train.T.dot(y_train)
    for i in range(0, num_iter):
        print(i)
        #estimating train error of gd
        pred = X_train.dot(W)
        Loss_sq = (pred - y_train) ** 2
        err_gd = np.sum(Loss_sq) / N
        err_gd_list.append(err_gd)
        gradient = a.dot(W) - b
        W =  W - alpha * gradient
    return W

def train_sgd_plot(X_train, y_train, alpha=0.1, reg=0, num_iter=10000, D=5000):
    ''' Build a model from X_train -> y_train using stochastic gradient descent '''
    global err_sgd_list
    N = X_train.shape[0]
    D = X_train.shape[1]
    W = np.ones([D, NUM_CLASSES])
    for i in range(num_iter):
        #estimating train error of sgd
        print(i)
        pred = X_train.dot(W)
        Loss_sq = (pred - y_train) ** 2
        err_sgd = np.sum(Loss_sq) / N
        err_sgd_list.append(err_sgd)
        u = np.random.randint(0, N)
        gradient = np.outer(X_train[u], W.T.dot(X_train[u]) - y_train[u]) + reg * W
        W = W - alpha * gradient
    return W
def train_sgd(X_train, y_train, alpha=0.1, reg=0, num_iter=10000, D=5000):
    ''' Build a model from X_train -> y_train using stochastic gradient descent '''

    N = X_train.shape[0]
    D = X_train.shape[1]
    W = np.ones([D, NUM_CLASSES])
    for i in range(num_iter):
        u = np.random.randint(0, N)
        gradient = np.outer(X_train[u], W.T.dot(X_train[u]) - y_train[u]) + reg * W
        W = W - alpha * gradient
    return W

def one_hot(labels_train):
    '''Convert categorical labels 0,1,2,....9 to standard basis vectors in R^{10} '''
    return np.array([np.eye(NUM_CLASSES)[ind] for ind in labels_train])

def predict(model, X):
    ''' From model and data points, output prediction vectors '''
    pred_labels_oh = X.dot(model)
    pred_labels = np.apply_along_axis(np.argmax, axis = 1, arr = pred_labels_oh)
    return pred_labels

def phi(X, G, b, D):
    ''' Featurize the inputs using random Fourier features '''
    # X.shape is n * P, where P is # of original feature
    # P is pixel features p = X.shape[1]
    # D is the new lifting features
    N = X.shape[0]
    B = np.tile(b, N).T
    XG = X.dot(G)
    entries = XG + B #gives a N * D ndarray
    cos_entries = np.sqrt(2) * np.cos(entries.flatten())
    phi = cos_entries.reshape((N, D))
    return phi
def call_closed_form():
    D =  5488
    VAR =  0.1
    alpha =  1e-08
    reg =  1e-05
    mean = np.zeros(D)
    cov = np.eye(D) * VAR
    G = np.random.multivariate_normal(mean, cov, P) # gives a P * D matrix
    b = np.random.uniform(0, 2 * np.pi, (D, 1)) # gives a D * 1 ndarray
    X_train_lift, X_test_lift = phi(X_train, G, b, D), phi(X_test, G, b, D) #X_train has been changed to X_train_lift, same as X_test
    
    model = train(X_train_lift, y_train, reg=0.9, D=D)
    
    pred_labels_train = predict(model, X_train_lift)
    pred_labels_test = predict(model, X_test_lift)
    model_closed = model
    print("Closed form solution ")
    print("D = ", D, ", sigma = ",VAR)
    print("alpha = ", alpha, ", reg = ", reg)
    print("Train accuracy: {0}".format(metrics.accuracy_score(labels_train, pred_labels_train)))
    print("Test accuracy: {0}".format(metrics.accuracy_score(labels_test, pred_labels_test)))

def call_gd(plot_err = False):
    D =  3920
    VAR =  0.1
    alpha =  1e-08
    reg =  2.5e-8
    mean = np.zeros(D)
    cov = np.eye(D) * VAR
    G = np.random.multivariate_normal(mean, cov, P)
    b = np.random.uniform(0, 2 * np.pi, (D, 1))
    X_train_lift, X_test_lift = phi(X_train, G, b, D), phi(X_test, G, b, D)
    if not plot_err:
        model = train_gd(X_train_lift, y_train, alpha, reg, num_iter=15000, D=D)
    else:
        model = train_gd_plot(X_train_lift, y_train, alpha, reg, num_iter=500, D=D)

    pred_labels_train = predict(model, X_train_lift)
    pred_labels_test = predict(model, X_test_lift)
    print("Batch gradient descent")
    print("D = ", D, ", sigma = ",VAR)
    print("alpha = ", alpha, ", reg = ", reg)
    print("Train accuracy: {0}".format(metrics.accuracy_score(labels_train, pred_labels_train)))
    print("Test accuracy: {0}".format(metrics.accuracy_score(labels_test, pred_labels_test)))

def call_sgd(plot_err = False):
    D =  3920
    VAR =  0.05
    alpha =  2e-05
    reg =  0.00078125
    mean = np.zeros(D)
    cov = np.eye(D) * VAR
    G = np.random.multivariate_normal(mean, cov, P) # gives a P * D matrix
    b = np.random.uniform(0, 2 * np.pi, (D, 1)) # gives a D * 1 ndarray
    X_train_lift, X_test_lift = phi(X_train, G, b, D), phi(X_test, G, b, D)
    if not plot_err:
        model = train_sgd(X_train_lift, y_train, alpha, reg, num_iter=60000, D=D)
    else:
        model = train_sgd_plot(X_train_lift, y_train, alpha, reg, num_iter=1000, D=D)

    pred_labels_train = predict(model, X_train_lift)
    pred_labels_test = predict(model, X_test_lift)
    print("Stochastic gradient descent")
    print("D = ", D, ", sigma = ",VAR)
    print("alpha = ", alpha, ", reg = ", reg)

if __name__ == "__main__":
    (X_train, labels_train), (X_test, labels_test) = load_dataset()
    y_train = one_hot(labels_train)
    y_test = one_hot(labels_test)

    P = X_train.shape[1]
    # call_closed_form()
    # call_gd()
    # call_sgd()
    # print("test done")

    # ID = np.array([i for i in range(X_test.shape[0])]).astype(int)
    # result = np.dstack((ID, pred_labels_test))[0].astype(int)
    # np.savetxt('test_label.csv', result, header = 'Id,Category', delimiter = ",")

    #error plotting
    err_gd_list = []
    err_sgd_list = []
    print("testing")
    call_sgd(plot_err = False)
    print(err_sgd_list)
    print("start plotting, sgd error rate")
    plt.figure(1)
    plt.plot(err_sgd_list, 'r-')
    plt.savefig("4-c")
    plt.clf()
    
    call_gd(plot_err = True)
    print("start plotting, gd error rate")
    plt.plot(err_gd_list, 'r-')
    plt.savefig("4-d")
    plt.clf()
    print("plot done")


