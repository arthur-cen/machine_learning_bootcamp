import numpy as np
from sklearn import linear_model, datasets
import matplotlib.pyplot as plt

# change the label to one hot vector
def onehot(y):
    # TODO: implement this
    # 1 0 0 0 0 0 0 0 0 0 --- 1
    # 0 1 0 0 0 0 0 0 0 0 --- 2
    # 0 0 1 0 0 0 0 0 0 0 --- 3
    # 0 0 0 1 0 0 0 0 0 0 --- 4
    # 0 0 0 0 1 0 0 0 0 0 --- 5
    # 0 0 0 0 0 1 0 0 0 0 --- 6
    # 0 0 0 0 0 0 1 0 0 0 --- 7
    # 0 0 0 0 0 0 0 1 0 0 --- 8
    # 0 0 0 0 0 0 0 0 1 0 --- 9
    # 0 0 0 0 0 0 0 0 0 1 --- 0 ...

    n = len(np.unique(y))
    m = y.shape[0]
    b = np.zeros((m, n))
    for i in xrange(m):
        b[i,y[i]] = 1
    return b

def softmax(X):
    expo = np.exp(X)
    return (expo.T / np.sum(expo, axis = 1)).T


def h_func(theta, X):
    # Adding the Bias Term
    h = np.dot(np.c_[np.ones(X.shape[0]), X], theta)
    return softmax(h)


def h_gradient(theta, X, y, lam=0.1):
    n = X.shape[0]
    y_mat = onehot(y)
    pred = h_func(theta, X)
    grad = - 1./n * np.dot(np.c_[np.ones(n), X].T, y_mat - pred) + lam * theta
                                                                # Added because of regularization
    return grad


def softmax_cost_func(theta, X, y, lam=0.1):
    n = X.shape[0]
    y_mat = onehot(y)

    cost = -1./n * np.sum(y_mat * np.log(h_func(theta, X))) + lam/2. * np.sum(theta * theta)
                # <This part is the cost function>           <This part is l2 regularization>
    return cost

# gradient descent
def softmax_grad_desc(theta, X, y, lr=.01, converge_change=.0001, max_iter=10000, lam=0.1):
    cost_iter = []
    cost = softmax_cost_func(theta, X, y)
    cost_iter.append([0, cost])
    cost_change = 1
    i = 1
    while cost_change > converge_change and i < max_iter:
        pre_cost = cost
        grad = h_gradient(theta, X, y, lam)
        # if (i % 50 == 0):
        #     lr /= 10
        theta -= lr * grad
        cost = softmax_cost_func(theta, X, y, lam)
        cost_change = abs(cost - pre_cost)
        cost_iter.append([i, cost])
        i+=1

    return theta, np.array(cost_iter)


def softmax_pred_val(theta, X):
    probability = h_func(theta, X)
    prediction = np.argmax(probability, axis=1) # take the one with highest probability as prediction
    return probability, prediction



def softmax_regression():
    digits = datasets.load_digits()
    # images_and_labels = list(zip(digits.images, digits.target))
    # for index, (image, lable) in enumerate(images_and_labels[:4]):
    #     plt.subplot(2, 4, index + 1)
    #     plt.axis('off')
    #     plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    #     plt.title('Training: %i' % lable)
    #     # uncomment to view the image
    #     # plt.show()
    n_samples = digits.images.shape[0]
    labels = digits.target
    X_train = digits.data[0:(n_samples / 4 * 3), :] # shape = (1347, 64)
    y_train = digits.target[0:(n_samples / 4 * 3), None]

    X_test = digits.data[n_samples / 4 * 3 ::, :] # shape = (450, 64)
    y_test = digits.target[n_samples / 4 * 3 ::, None]
    #self-implemented softmax regression classifier
    theta = np.random.rand(X_train.shape[1] + 1, len(np.unique(y_train)))
    fit_val, cost_iter = softmax_grad_desc(theta, X_train, y_train)
    _, train_pred = softmax_pred_val(theta, X_train)
    _, test_pred = softmax_pred_val(theta, X_test)

    print(cost_iter[-1,:])
    print('Accuracy of train: {}'.format(np.mean(train_pred[:, None] == y_train)))
    print('Accuracy of validation: {}'.format(np.mean(test_pred[:, None] == y_test)))

    plt.plot(cost_iter[:, 0], cost_iter[:, 1])
    plt.ylabel("Cost")
    plt.xlabel("Iteration")
    plt.show()

    #sklearn softmax regression classifier

    #print(fitted_val)
    # print(cost_iter[-1,:])
    # print('Accuracy: {}'.format(np.mean(preds[:, None] == y)))
    #
    # plt.plot(cost_iter[:, 0], cost_iter[:, 1])
    # plt.ylabel("Cost")
    # plt.xlabel("Iteration")
    # plt.show()


def main():
    softmax_regression()

if __name__ == "__main__":
    main()