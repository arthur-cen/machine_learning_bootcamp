import numpy as np

class MLP:
    def __init__(self, input_size, output_size, hidden_layer_size=[100], batch_size=200, activation="sigmoid", output_layer='softmax', loss='cross_entropy', lr=0.01, reg_lambda=0.0001, momentum=0.9, verbose=10):
        """
        Multilayer perceptron Class
        :param input_size: int, input size (n_feature)
        :param output_size: int,  output node size (n_class)
        :param hidden_layer_size: list of int, each int represents the size of a hidden layer
        :param batch_size: int, batch_size
        :param activation: string, activation function ['sigmoid', 'tanh']
        :param output_layer: string, output layer type ['softmax']
        :param loss: string, loss type ['cross_entropy']
        :param lr: float, learning rate
        :param reg_lambda: float, lambda of regularization
        :param verbose: int, print flag
        :param momentum: float, momentum
        """
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layer_size = hidden_layer_size
        self.lr = lr
        self.reg_lambda = reg_lambda
        self.momentum = momentum
        self.batch_size = batch_size
        self.n_layers = len(hidden_layer_size) # only hidden layers
        self.activation = activation
        self.verbose = verbose

        if activation == 'sigmoid':
            self.activation_func = sigmoid
            self.activation_dfunc = dsigmoid
        elif activation == 'tanh':
            self.activation_func = tanh
            self.activation_dfunc = dtanh
        else:
            raise ValueError("Currently only supoort 'sigmoid' or 'tanh' activations func!")

        self.loss = loss
        if output_layer == 'softmax':
            self.output_layer = softmax
        else:
            raise ValueError('Currently only Support softmax output_layer!')

        self.nclass = output_size

        self.weights = []   # store weights
        self.bias = []      # store bias
        self.layers = []    # store forwarding activation values
        self.deltas = []    # store errors for backprop

    def get_weight_bound(self, fan_in, fan_out):
        """
        Generate bound value for random weights initialization
        :param fan_in: layer input size
        :param fan_out: layer output size
        :return: float, bound
        """
        if self.activation == 'sigmoid':
            init_bound = np.sqrt(2. / (fan_in + fan_out))
        elif self.activation == 'tanh':
            init_bound = np.sqrt(6. / (fan_in + fan_out))
        return init_bound

    def fit(self, X, y, max_epochs, shuffle_data):
        """
        fit the model given data X and label y
        :param X: array-like, shape(n_samples, n_features)
        :param y: array-like, shape(n_samples, 1)
        :param max_epochs: int, max iterations
        :param shuffle_data: bool, if shuffle the data.
        :return: MLP model object
        """
        # print("Input X's shape is: {}".format(X.shape)) #X.shape = (1500, 64)
        # print("Input y's shape is: {}".format(y.shape)) #y.shape = (1500,)

        n_samples, n_features = X.shape
        if y.shape[0] != n_samples:
            raise ValueError("Shapes of X and y don't fit!")

        # generate weights
        # Weights and bias connecting input layer and first hidden layer
        init_bound = self.get_weight_bound(n_features, self.hidden_layer_size[0])
        self.weights.append(np.random.uniform(-init_bound, init_bound, size=(n_features, self.hidden_layer_size[0])))
        self.bias.append(np.random.uniform(-init_bound, init_bound, self.hidden_layer_size[0]))

        # Weights and bias connecting hidden layers
        for i in range(1, len(self.hidden_layer_size)):
            init_bound = self.get_weight_bound(self.hidden_layer_size[i-1], self.hidden_layer_size[i])
            self.weights.append(np.random.uniform(-init_bound, init_bound, size=(self.hidden_layer_size[i-1], self.hidden_layer_size[i])))
            self.bias.append(np.random.uniform(-init_bound, init_bound, self.hidden_layer_size[i]))

        # Weights and bias connecting last hidden layer and output layer
        init_bound = self.get_weight_bound(self.hidden_layer_size[-1], self.output_size)
        self.weights.append(np.random.uniform(-init_bound, init_bound, size=(self.hidden_layer_size[-1], self.output_size)))
        self.bias.append(np.random.uniform(-init_bound, init_bound, self.output_size))

        # pre-allocate memory for both activations and errors
        # for input layer
        self.layers.append(np.empty((self.batch_size, self.input_size)))
        # for hidden layers
        for i in range(0, self.n_layers):
            self.layers.append(np.empty((self.batch_size, self.hidden_layer_size[i])))
            self.deltas.append(np.empty((self.batch_size, self.hidden_layer_size[i]))) #coresponding delta and layer has equal size
        # for output layer
        self.layers.append(np.empty((self.batch_size, self.output_size)))
        self.deltas.append(np.empty((self.batch_size, self.output_size))) #
        # print("layers number = {}".format(len(self.layers))) #5
        # print("deltas number = {}".format(len(self.deltas))) #4
        # print("weights number = {}".format(len(self.weights))) #4
        # print("bias number = {}".format(len(self.bias))) #4

        # main loop
        for i in xrange(max_epochs):

            # shuffle data
            if shuffle_data:
                # Using sklearn.utils.shuffle
                shuffle(X, y)

            # iterate every batch
            for batch in xrange(0, n_samples, self.batch_size):
                #call forward function
                self.forward(X)
                #call backward function
                self.backward(X, y)

            if self.verbose == 0:
                # Compute Loss and Training Accuracy
                loss = self.compute_loss(X, y)
                acc = self.score(X, y)
                print('Epoch {}: loss = {}, accuracy = {}'.format(i, loss, acc))

        return self

    def compute_loss(self, X, y):
        """
        Compute loss
        :param X: data, array-like, shape(n_sample, n_feature)
        :param y: label, array-like, shape(n_sample, 1)
        :return: loss value
        """
        n_samples = X.shape[0]
        probs = self.forward(X)

        # Calculating the loss
        logprob = -np.log(probs[range(n_samples, y)])
        data_loss = np.sum(logprob, axis=0)
        print(data_loss.shape)
        # Add regularization term to loss
        data_loss += self.reg_lambda / 2 * np.sum(np.array([np.dot(w.ravel(), w.ravel()) for w in self.weights])) #TODO put this into cheat sheet
                                                                                                                   #TODO and write out the mathematic/matrix formula
        return 1. / n_samples * data_loss

    def forward(self, X):
        # input layer
        self.layers[0] = X
        hid_layer = X
        # print("hiddien layer size = {}".format(self.hidden_layer_size))
        # print("weight array length = {}".format(len(self.weights)))
        # layer list size should be 1 larger than weight list size
        # hidden layers
        # Investigate layer and weight sizes
        # for i in range(len(self.layers)):
        #     print("layer #{0} shape = {1}".format(i, self.layers[i].shape))
        # for i in range(len(self.weights)):
        #     print("weights #{0} shape = {1}".format(i, self.weights[i].shape))

        for i in xrange(0, len(self.weights) - 1):
            # Calculate layer output z_i
            z_i = np.dot(self.layers[i], self.weights[i])
            # Adding Bias Term bias_i to layer output z_i
            z_i += self.bias[i] #TODO need clarification here
            # Activate neurons in layer i !
            a_i = self.activation_func(z_i)
            # Assign the activated output to the next layer
            self.layers[i + 1] = a_i

        # output layer (Note here the activation is using output_layer func)
        z_out = np.dot(self.layers[-2], self.weights[-1])
        z_out += self.bias[-1]
        a_out = self.output_layer(z_out)
        self.layers[-1] = a_out

        return self.layers[-1]
    def backward_prop(self, X, y):
        def backward(self, X, y):
            if self.loss == 'cross_entropy':
                self.deltas[-1] = self.layers[-1]
                # cross_entropy loss backprop
                self.deltas[-1][range(X.shape[0]), y] -= 1  # TODO figure out why??? Related to One hot encoding?

            # TODO update deltas
            for i in xrange(self.n_layers, 0, -1):
                # print(self.deltas[i].shape)
                # print(self.weights[i].shape)
                # print(self.activation_func(self.layers[i]).shape)
                # quit()
                self.deltas[i - 1] = np.dot(self.deltas[i], self.weights[i].T) * self.activation_dfunc(self.layers[i])

            # TODO update weights
            for i in xrange(len(self.layers) - 1, -1, -1):
                # This is the derivative of  regulizer term
                grad = self.layers[i].T.dot(self.deltas[i]) + self.reg_lambda * self.weights[
                    i] / self.batch_size  # TODO write this update code into cheat sheet
                self.weights[i] -= self.lr * grad
                self.bias[i] -= self.lr * np.mean(self.deltas[i], axis=0)

    def predict(self, X):
        """
        predicting probability outputs
        :param X: array-like, shape(n_samples, n_features)
        :return: array-like, predicted probabilities
        """
        return self.forward(X)

    def score(self, X, y):
        """
        compute accuracy
        :param X: array-like, shape(n_samples, n_features)
        :param y: ground truth labels array-like, shape(n_samples, 1)
        :return: float, accuracy
        """
        n_samples = X.shape[0]
        pred = np.argmax(self.predict(X))
        acc = np.sum(pred == y) / n_samples
        return acc



