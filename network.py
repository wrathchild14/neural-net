import pandas as pd
import numpy as np
import pickle


class Network(object):
    def __init__(self, sizes, optimizer="sgd"):
        # weights connect two layers, each neuron in layer L is connected to every neuron in layer L+1,
        # the weights for that layer of dimensions size(L+1) X size(L)
        # the bias in each layer L is connected to each neuron in L+1, the number of weights necessary for the bias
        # in layer L is therefore size(L+1).
        # The weights are initialized with a He initializer: https://arxiv.org/pdf/1502.01852v1.pdf
        self.weights = [((2 / sizes[i - 1]) ** 0.5) * np.random.randn(sizes[i], sizes[i - 1]) for i in
                        range(1, len(sizes))]
        self.biases = [np.zeros((x, 1)) for x in sizes[1:]]
        self.optimizer = optimizer
        self.l2_lambda = 0.01
        if self.optimizer == "adam":
            # Implement the buffers necessary for the Adam optimizer.
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.adam_eps = 1e-8
            self.mean_w = [np.zeros_like(w) for w in self.weights]
            self.variance_w = [np.zeros_like(w) for w in self.weights]
            self.mean_b = [np.zeros_like(b) for b in self.biases]
            self.variance_b = [np.zeros_like(b) for b in self.biases]
        print(f"Log: initialized network with the {self.optimizer} optimizer")

    def train(self, training_data, training_class, eval_data, eval_class, epochs, mini_batch_size, learning_rate,
              decay_rate, enable_l2):
        # training data - numpy array of dimensions [n0 x m], where m is the number of examples in the data and
        # n0 is the number of input attributes
        # training_class - numpy array of dimensions [c x m], where c is the number of classes
        # epochs - number of passes over the dataset
        # mini_batch_size - number of examples the network uses to compute the gradient estimation

        iteration_index = 0
        learning_rate_current = learning_rate

        n = training_data.shape[1]
        for j in range(epochs):
            print("Epoch" + str(j))
            loss_avg = 0.0
            mini_batches = [
                (training_data[:, k:k + mini_batch_size], training_class[:, k:k + mini_batch_size])
                for k in range(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
                output_activation, zs, activations = self.forward_pass(mini_batch[0])
                gw, gb = net.backward_pass(output_activation, mini_batch[1], zs, activations)

                self.update_network(gw, gb, learning_rate_current, iteration_index)

                # Exponential learning rate decay
                learning_rate_current = learning_rate * np.exp(-decay_rate * j)
                iteration_index += 1
                loss = cross_entropy(mini_batch[1], output_activation)
                # L2 regularisation loss
                if enable_l2:
                    loss += (self.l2_lambda / (2 * mini_batch_size)) * sum([np.sum(np.square(w)) for w in self.weights])

                loss_avg += loss

            print("Epoch {} complete".format(j))
            print("Loss:" + str(loss_avg / len(mini_batches)))
            if j % 10 == 0:
                self.eval_network(eval_data, eval_class)

    def eval_network(self, validation_data, validation_class):
        # validation data - numpy array of dimensions [n0 x m], where m is the number of examples in the data and
        # n0 is the number of input attributes
        # validation_class - numpy array of dimensions [c x m], where c is the number of classes
        n = validation_data.shape[1]
        loss_avg = 0.0
        tp = 0.0
        for i in range(validation_data.shape[1]):
            example = np.expand_dims(validation_data[:, i], -1)
            example_class = np.expand_dims(validation_class[:, i], -1)
            example_class_num = np.argmax(validation_class[:, i], axis=0)
            output, zs, activations = self.forward_pass(example)
            output_num = np.argmax(output, axis=0)[0]
            tp += int(example_class_num == output_num)

            loss = cross_entropy(example_class, output)
            loss_avg += loss
        print("Validation Loss:" + str(loss_avg / n))
        print("Classification accuracy: " + str(tp / n))

    def update_network(self, gw, gb, eta, iteration_index):
        # gw - weight gradients - list with elements of the same shape as elements in self.weights
        # gb - bias gradients - list with elements of the same shape as elements in self.biases
        # eta - learning rate
        # SGD
        if self.optimizer == "sgd":
            for i in range(len(self.weights)):
                self.weights[i] -= eta * gw[i]
                self.biases[i] -= eta * gb[i]
        elif self.optimizer == "adam":
            for i in range(len(self.weights)):
                self.mean_w[i] = self.beta1 * self.mean_w[i] + (1 - self.beta1) * gw[i]
                self.variance_w[i] = self.beta2 * self.variance_w[i] + (1 - self.beta2) * gw[i] * gw[i]
                mean_w_hat = self.mean_w[i] / (1 - self.beta1 ** (iteration_index + 1))
                variance_w_hat = self.variance_w[i] / (1 - self.beta2 ** (iteration_index + 1))
                self.weights[i] -= eta * mean_w_hat / (np.sqrt(variance_w_hat) + self.adam_eps)

                self.mean_b[i] = self.beta1 * self.mean_b[i] + (1 - self.beta1) * gb[i]
                self.variance_b[i] = self.beta2 * self.variance_b[i] + (1 - self.beta2) * gb[i] * gb[i]
                mean_b_hat = self.mean_b[i] / (1 - self.beta1 ** (iteration_index + 1))
                variance_b_hat = self.variance_b[i] / (1 - self.beta2 ** (iteration_index + 1))
                self.biases[i] -= eta * mean_b_hat / (np.sqrt(variance_b_hat) + self.adam_eps)
        else:
            raise ValueError('Unknown optimizer:' + self.optimizer)

    def forward_pass(self, x):
        # input - numpy array of dimensions [n0 x m], where m is the number of examples in the mini batch and
        # n0 is the number of input attributes
        zs = []
        activation = x
        activations = [x]
        for i in range(len(self.weights) - 1):
            z = np.dot(self.weights[i], activation) + self.biases[i]
            zs.append(z)

            activation = sigmoid(z)
            activations.append(activation)

        z = np.dot(self.weights[-1], activation) + self.biases[-1]
        zs.append(z)

        activation = softmax(z)
        activations.append(activation)

        return activation, zs, activations

    def backward_pass(self, output, target, zs, activations):
        delta = softmax_dl_dz(output, target)

        nabla_w = [np.dot(delta, activations[-2].T)]
        nabla_b = [np.sum(delta, axis=1, keepdims=True)]

        for L in range(2, len(self.weights) + 1):
            z = zs[-L]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-L + 1].T, delta) * sp
            nabla_w.insert(0, np.dot(delta, activations[-L - 1].T))
            nabla_b.insert(0, np.sum(delta, axis=1, keepdims=True))

        return nabla_w, nabla_b


def softmax(z):
    exp_z = np.exp(z - np.max(z))
    return exp_z / exp_z.sum(axis=0, keepdims=True)


def softmax_dl_dz(output, target):
    # partial derivative of the cross entropy loss w.r.t Z at the last layer
    return output - target


def cross_entropy(y_true, y_predicted, epsilon=1e-12):
    targets = y_true.transpose()
    predictions = y_predicted.transpose()
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    n = predictions.shape[0]
    ce = -np.sum(targets * np.log(predictions + 1e-9)) / n
    return ce


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


def unpickle(file):
    with open(file, 'rb') as fo:
        return pickle.load(fo, encoding='bytes')


def load_data_cifar(train_file, test_file):
    train_dict = unpickle(train_file)
    test_dict = unpickle(test_file)
    train_data = np.array(train_dict['data']) / 255.0
    train_class = np.array(train_dict['labels'])
    train_class_one_hot = np.zeros((train_data.shape[0], 10))
    train_class_one_hot[np.arange(train_class.shape[0]), train_class] = 1.0
    test_data = np.array(test_dict['data']) / 255.0
    test_class = np.array(test_dict['labels'])
    test_class_one_hot = np.zeros((test_class.shape[0], 10))
    test_class_one_hot[np.arange(test_class.shape[0]), test_class] = 1.0
    return train_data.transpose(), train_class_one_hot.transpose(), test_data.transpose(), test_class_one_hot.transpose()


if __name__ == "__main__":
    train_file = "./data/train_data.pckl"
    test_file = "./data/test_data.pckl"
    train_data, train_class, test_data, test_class = load_data_cifar(train_file, test_file)
    val_pct = 0.1
    val_size = int(len(train_data) * val_pct)
    val_data = train_data[..., :val_size]
    val_class = train_class[..., :val_size]
    train_data = train_data[..., val_size:]
    train_class = train_class[..., val_size:]
    # The Network takes as input a list of the numbers of neurons at each layer. The first layer has to match the
    # number of input attributes from the data, and the last layer has to match the number of output classes
    # The initial settings are not even close to the optimal network architecture, try increasing the number of layers
    # and neurons and see what happens.
    net = Network([train_data.shape[0], 200, 10], optimizer="adam")
    # epoch, batch_size, learning_rate, decay_rate, l2_lambda = 20, 16, 0.01, 0.05, 0.001
    epochs, batch_size, learning_rate, decay_rate, enable_l2 = 50, 16, 2e-5, 0.001, True
    net.train(train_data, train_class, val_data, val_class, epochs, batch_size, learning_rate, decay_rate, enable_l2)
    net.eval_network(test_data, test_class)
    print(f"Log: finished with params: epochs - {epochs}, batch size - {batch_size}, learning rate - {learning_rate},"
          f" decay rate - {decay_rate}, lambda for L2 - {0.01}")
