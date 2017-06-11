import numpy as np
from multilayer_perceptron_classifier import *

def main():
    def my_mlp():
        # from sklearn.datasets import fetch_mldata
        # mnist = fetch_mldata("MNIST original")
        # X, y = mnist.data / 255., mnist.target
        # X_train, X_test = X[:60000], X[60000:]
        # y_train, y_test = y[:60000], y[60000:]

        import sklearn.datasets
        dataset = sklearn.datasets.load_digits()
        X_train = dataset.data[:1500]
        X_test = dataset.data[1500:]
        y_train = dataset.target[:1500]
        y_test = dataset.target[1500:]

        network = MLP(input_size=64, output_size=10, hidden_layer_size=[128, 64, 32], batch_size=200,
                      activation="sigmoid",
                      output_layer='softmax', loss='cross_entropy', lr=0.1)

        network.fit(X_train, y_train, 100, True)

        acc = network.score(X_test, y_test)
        print('Test Accuracy: {}'.format(acc))

    def sklearn_mlp():
        import matplotlib.pyplot as plt
        from sklearn.datasets import fetch_mldata
        from sklearn.neural_network import MLPClassifier

        # mnist =fetch_mldata("MNIST original")
        # X, y = mnist.data / 255., mnist.target
        # X_train, X_test = X[:60000], X[60000:]
        # y_train, y_test = y[:60000], y[60000:]

        import sklearn.datasets
        dataset = sklearn.datasets.load_digits()
        X_train = dataset.data[:1500]
        X_test = dataset.data[1500:]
        y_train = dataset.target[:1500]
        y_test = dataset.target[1500:]

        mlp = MLPClassifier(hidden_layer_sizes=(128), max_iter=100, alpha=1e-4,
                            solver='sgd', activation='logistic', verbose=10, tol=1e-4, random_state=1,
                            learning_rate_init=.01)
        mlp.fit(X_train, y_train)
        print("Training set score: %f" % mlp.score(X_train, y_train))
        print("Test set score: %f" % mlp.score(X_test, y_test))

    def main():
        print('Class 2 Multiple Layer Perceptron (MLP) Example')
        my_mlp()

        print ('')

        print('Class 2 sklearn MLP Example')
        sklearn_mlp()

if __name__ == "__main__":
    main()
