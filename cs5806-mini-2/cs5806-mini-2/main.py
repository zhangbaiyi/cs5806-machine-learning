import random

import numpy as np

np.random.seed(5806)
import pandas as pd
# pd.set_option('float_format', '{:.2f}'.format)
import matplotlib.pyplot as plt
import math


class Input:
    def __init__(self, data):
        """
        Initializes the Input class.
        :param data: List, the input data.
        :return: None
        """
        df = pd.DataFrame(data)
        if df.isnull().values.any():
            raise ValueError("The input data contains missing values.")
        self.observation = df.shape[0]
        self.data = df


class MultiLayerPerceptron:
    def __init__(self, nLayers, arch, batch_size=10, lr=0.01, epochs=10000, sse_threshold=0.02, a_function="tanh",
                 momentum=0.0):
        """
        Initializes the MLP based on the user-defined architecture.
        :param nLayers: Integer, the number of layers in the MLP.
        :param arch: List of integers, the number of neurons in each layer.
        :param input: Input, the input data.
        :return: A tuple of two lists, containing the initialized weights and biases for each layer.
        """
        weights = []
        biases = []
        if len(arch) != nLayers:
            raise ValueError("The number of layers in the architecture must match the number of layers specified.")
        layer_sizes = [1] + arch
        for i in range(1, len(layer_sizes)):
            weights.append(
                np.random.randn(layer_sizes[i], layer_sizes[i - 1]))
            biases.append(np.random.randn(layer_sizes[i], 1))

        self.weights = weights
        self.biases = biases
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.a_function = a_function
        self.sse_threshold = sse_threshold
        self.sse_per_iteration = []
        self.momentum = momentum
        self.prev_deltas_w = [np.zeros(w.shape) for w in self.weights]
        self.prev_deltas_b = [np.zeros(b.shape) for b in self.biases]

    def feedforward(self, i):
        """
        Performs forward propagation to compute the output of the MLP.
        :param input: Input, the input data.
        :return: List, the output of the MLP.
        """
        o = []
        for p in i:
            a = p
            for i in range(len(self.weights) - 1):
                # use sigmoid
                # use @ instead of
                a = activate(np.dot(self.weights[i], a) + self.biases[i], type=self.a_function)
            a = purelin(np.dot(self.weights[-1], a) + self.biases[-1])
            o.append(a.flatten()[0])
        return o

    def SGD(self, input, target):
        """
        Stochastic Gradient Descent algorithm to train the MLP.
        :param input: Input, the input data.
        :return: None
        """
        training_data = list(zip(input, target))
        # df = pd.concat([input.data, pd.DataFrame(target)], axis=1)
        # training_data = list(df.itertuples(index=False, name=None))
        random.shuffle(training_data)
        mini_batches = [training_data[k:k + self.batch_size] for k in range(0, len(training_data), self.batch_size)]
        for i in range(self.epochs):
            for mini_batch in mini_batches:
                # self.update_mini_batch(mini_batch)
                for x, y in mini_batch:
                    self.bprop(x, y)
            errors = []
            response = self.feedforward(input)
            response = np.array(response).flatten()
            targ = np.array(target).flatten()
            for j in range(len(response)):
                errors.append((response[j] - targ[j]) ** 2)
            self.sse_per_iteration.append(round(np.sum(errors), 2))
            # print(self.sse_per_iteration[-1])
            if len(self.sse_per_iteration) > 0 and self.sse_per_iteration[-1] <= self.sse_threshold:
                break
            if i % 1000 == 0:
                print(f"Epoch {i} complete.")

    def bprop(self, x, y):
        """

        """
        # Feedforward
        n_list = []
        a = x
        a_list = [a]
        for i in range(len(self.weights) - 1):
            n = np.dot(self.weights[i], a) + self.biases[i]
            n_list.append(n.flatten())
            a = activate(n, type=self.a_function)
            a_list.append(a)
        n = np.dot(self.weights[-1], a) + self.biases[-1]
        n_list.append(n.flatten())
        a = purelin(n)
        a_list.append(a)
        # for w, b in zip(self.weights, self.biases):
        #     n = np.dot(w, a) + b
        #     n_list.append(n.flatten())
        #     a = purelin(n)
        #     a_list.append(a)
        error = self.error(a, y).flatten()
        # Backpropagation
        s = (-2 * purelin_prime(n_list[-1]) * error)[0]  # s for the output layer
        s_list = [s]
        deltas_w = [np.array(self.lr * np.dot(s, a_list[-2].T)).reshape(self.weights[-1].shape)]
        deltas_b = [np.array(self.lr * s).reshape(self.biases[-1].shape)]

        for l in range(2, len(self.weights) + 1):
            s = np.dot(np.diag(activate_prime(n_list[-l], type=self.a_function).flatten()),
                       np.dot(self.weights[-l + 1].T, s_list[-l + 1]))
            s_list = [s] + s_list
            # delta_w = np.array(self.lr * np.dot(s, a_list[-l - 1].T)).reshape(self.weights[-l].shape)
            # delta_b = np.array(self.lr * s).reshape(self.biases[-l].shape)
            delta_w = np.array(self.lr / self.batch_size * np.dot(s, a_list[-l - 1].T)).reshape(self.weights[-l].shape)
            delta_b = np.array(self.lr / self.batch_size * s).reshape(self.biases[-l].shape)
            # delta_w = np.array(self.lr * np.dot(s_list[-l], np.array(a_list[-l - 1]).T)).reshape(self.weights[-l].shape)
            # delta_b = np.array(self.lr * s_list[-l]).reshape(self.biases[-l].shape)
            deltas_w = [delta_w] + deltas_w
            deltas_b = [delta_b] + deltas_b
            # delta_w = np.array(self.lr * np.dot(s, np.array(a_list[-l - 1]).T)).reshape(self.weights[-l].shape)
            # delta_b = np.array(self.lr * s).reshape(self.biases[-l].shape)
            # deltas_w = [delta_w] + deltas_w
            # deltas_b = [delta_b] + deltas_b

        for i in range(len(self.weights)):
            self.weights[i] -= (self.prev_deltas_b[i] * self.momentum + (1 - self.momentum) * deltas_w[i])
            self.biases[i] -= (self.prev_deltas_b[i] * self.momentum + (1 - self.momentum) * deltas_b[i])
            # self.biases[i] -= deltas_b[i]

        self.prev_deltas_b = deltas_b
        self.prev_deltas_w = deltas_w

    def error(self, output_activations, y):
        return np.round(y - output_activations, 4)
        return y - output_activations


def activate(x, type="tanh"):
    """
    Activation function. Hyperbolic tangent function in this program
    :param x: z
    :param type: type of activation function
    :return: activated z
    """
    if type == "tanh":
        return np.tanh(x)
    elif type == "sigmoid":
        return 1 / (1 + np.exp(-x))
    elif type == "relu":
        return np.maximum(0, x)


def purelin(x):
    return x


def purelin_prime(x):
    return np.ones(x.shape)


def activate_prime(x, type="tanh"):
    """
    Derivative of the activation function.
    :param x: z
    :param type: type of activation function
    :return: derivative of the activation function
    """
    if type == "tanh":
        return 1 - np.tanh(x) ** 2
    elif type == "sigmoid":
        return np.exp(-x) / (1 + np.exp(-x)) ** 2
    elif type == "relu":
        return np.where(x > 0, 1, 0)


def parse_input(input_str):
    """
    Parses the input string to handle 'pi' and numeric values.
    Returns the numeric value of the input.
    """
    if 'pi' in input_str:
        try:
            # Replace 'pi' with 'math.pi' and evaluate the expression
            input_str = input_str.replace('pi', 'math.pi')
            return eval(input_str)
        except:
            print("Invalid input. Please make sure to use 'pi' for pi and include numbers correctly.")
            return None
    else:
        try:
            # Directly convert numeric input to float
            return float(input_str)
        except ValueError:
            print("Invalid input. Please enter a valid number.")
            return None


def mlp_init_prompt():
    """
    Prompts the user to input the number of layers and the architecture as well as input of the MLP.
    :return: Tuple of two integers, the number of layers and the architecture as well as input of the MLP.
    """
    print("Please enter the number of layers in the MLP:")
    nLayers = int(input())
    print("Please enter the architecture of the MLP:")
    arch = list(map(int, input().split()))
    if len(arch) != nLayers:
        raise ValueError("The number of layers in the architecture must match the number of layers specified.")
    print("Please enter the input observations:")
    input_obs = int(input())
    print("Please enter the input function:\n 1: sin(p), 2: p^2, 3: e^p, 4: sin(p)^2 + cos(p)^3")
    f = int(input())
    print("Please enter the input range: [a,b]")
    print("Please enter a: (Use 'pi' to represent pi, -2pi should be entered as -2*pi)")
    a = parse_input(input())
    print("Please enter b: (Use 'pi' to represent pi, 2pi should be entered as 2*pi)")
    b = parse_input(input())
    if a is None or b is None:
        raise ValueError("Invalid input. Please make sure to enter valid numbers.")
    if f == 1:
        input_data = [[math.sin(x)] for x in np.linspace(a, b, input_obs)]
    elif f == 2:
        input_data = [[x ** 2] for x in np.linspace(a, b, input_obs)]
    elif f == 3:
        input_data = [[math.exp(x)] for x in np.linspace(a, b, input_obs)]
    elif f == 4:
        input_data = [[math.sin(x) ** 2 + math.cos(x) ** 3] for x in np.linspace(a, b, input_obs)]
    else:
        raise ValueError("Invalid input function. Please make sure to enter a valid function.")
    input_data = pd.DataFrame(input_data)
    i = Input(input_data)
    return nLayers, arch, i


def report_plot_with_momentum(p, target, network_output, sse_per_iteration, t_label="Target", nLayers=0, arch=[], learning_rate=0.01, momentum=0.0):
    """
    Plots the target and response functions.
    :param p: Input, the input data.
    :param target: List, the target function values.
    :param network_output: List, the network output.
    :param sse_per_iteration: List, the sum of squared errors per iteration.
    :return: None
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].plot(p, target, 'ro', label=t_label)
    axs[0].plot(p, network_output, 'g-', label='Network output')
    axs[0].set_title("Number of iterations: " + str(len(sse_per_iteration)) + "\n" + "Number of layers: " + str(
        nLayers) + "\n" + "Architecture: " + str(arch))
    axs[0].set_xlabel('p')
    axs[0].set_ylabel('Magnitude')
    axs[0].grid()
    axs[0].legend()

    iterations = range(1, len(sse_per_iteration) + 1)
    axs[1].loglog(iterations, sse_per_iteration, marker='o', linestyle='-')
    axs[1].set_title("SSE: " + str(sse_per_iteration[-1]) + "\n" + "Learning rate: " + str(
        learning_rate) + "\n" + "Number of iterations: " + str(len(sse_per_iteration)) + "\n" + "Architecture: " + str(
        arch) + "\n" + "SSE threshold: " + str(0.01) + "\n" + "Momentum: " + str(momentum))
    axs[1].set_xlabel('Number of iterations (log scale)')
    axs[1].set_ylabel('Magnitude (log scale)')
    axs[1].grid()
    axs[1].legend(['SSE'])

    plt.tight_layout()
    plt.show()

def report_plot(p, target, network_output, sse_per_iteration, t_label="Target", nLayers=0, arch=[],
                learning_rate=0.01):
    """
    Plots the target and response functions.
    :param p: Input, the input data.
    :param target: List, the target function values.
    :param network_output: List, the network output.
    :param sse_per_iteration: List, the sum of squared errors per iteration.
    :return: None
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].plot(p, target, 'ro', label=t_label)
    axs[0].plot(p, network_output, 'g-', label='Network output')
    axs[0].set_title("Number of iterations: " + str(len(sse_per_iteration)) + "\n" + "Number of layers: " + str(
        nLayers) + "\n" + "Architecture: " + str(arch))
    axs[0].set_xlabel('p')
    axs[0].set_ylabel('Magnitude')
    axs[0].grid()
    axs[0].legend()

    iterations = range(1, len(sse_per_iteration) + 1)
    axs[1].loglog(iterations, sse_per_iteration, marker='o', linestyle='-')
    axs[1].set_title("SSE: " + str(sse_per_iteration[-1]) + "\n" + "Learning rate: " + str(
        learning_rate) + "\n" + "Number of iterations: " + str(len(sse_per_iteration)) + "\n" + "Architecture: " + str(
        arch) + "\n" + "SSE threshold: " + str(0.01))
    axs[1].set_xlabel('Number of iterations (log scale)')
    axs[1].set_ylabel('Magnitude (log scale)')
    axs[1].grid()
    axs[1].legend(['SSE'])

    plt.tight_layout()
    plt.show()


def q1_plt_target_and_response(input, target, response):
    """
    Plots the target and response functions.
    :param input: Input, the input data.
    :param target: List, the target function values.
    :return: None
    """
    plt.plot(input, target, label='Sinusoidal function')
    plt.plot(input, response, label='Network response')
    plt.xlabel('p')
    plt.ylabel('Magnitude')
    plt.xlim(-7, 7)
    plt.title('Untrained Network Response vs. Sinusoidal Target Function')
    plt.legend()
    plt.grid()
    plt.show()


def q1():
    # pass
    # nLayers, arch, input = mlp_init_prompt()
    nLayers = 4
    arch = [2, 4, 3, 1]
    input_data = [x for x in np.linspace(-2 * math.pi, 2 * math.pi, 100)]
    mlp = MultiLayerPerceptron(nLayers, arch)
    target = [math.sin(x) for x in input_data]
    response = mlp.feedforward(input_data)
    q1_plt_target_and_response(input_data, target, response)
    print("Weights and biases initialized:")
    for i in range(len(mlp.weights)):
        print(f"Layer {i + 1} weights (dim: {mlp.weights[i].shape}):\n {mlp.weights[i]}")
        print(f"Layer {i + 1} biases (dim: {mlp.biases[i].shape}):\n {mlp.biases[i]}")


def q2_comp():
    nLayers = 3
    arch = [10, 5, 1]
    input_data = [x for x in np.linspace(-2 * math.pi, 2 * math.pi, 100)]
    target = [math.sin(x) ** 2 + math.cos(x) ** 3 for x in input_data]
    mlp = MultiLayerPerceptron(nLayers, arch, batch_size=10, lr=0.02, epochs=200000, sse_threshold=0.01,
                               a_function="tanh")
    mlp.SGD(input_data, target)
    response = mlp.feedforward(input_data)
    report_plot(input_data, target, response, mlp.sse_per_iteration, t_label='Sinusoidal function', nLayers=nLayers,
                arch=arch, learning_rate=0.02)


def q2_exp():
    nLayers = 2
    arch = [3, 1]
    input_data = [x for x in np.linspace(0, 2, 100)]
    target = [math.exp(x) for x in input_data]
    mlp = MultiLayerPerceptron(nLayers, arch, batch_size=10, lr=0.02, epochs=200000, sse_threshold=0.01,
                               a_function="tanh")
    mlp.SGD(input_data, target)
    response = mlp.feedforward(input_data)
    report_plot(input_data, target, response, mlp.sse_per_iteration, t_label='Exponential function', nLayers=nLayers,
                arch=arch, learning_rate=0.02)


def q2_sin():
    nLayers = 2
    arch = [7, 1]
    input_data = [x for x in np.linspace(-2 * math.pi, 2 * math.pi, 100)]

    target = np.array(np.sin(input_data)).tolist()
    mlp = MultiLayerPerceptron(nLayers, arch, batch_size=100, lr=0.02, epochs=200000, sse_threshold=0.01,
                               a_function="sigmoid")
    print("Initial weights and biases:")
    print(mlp.weights)
    print(mlp.biases)
    mlp.SGD(input_data, target)
    print("Final weights and biases:")
    print(mlp.weights)
    print(mlp.biases)
    response = mlp.feedforward(input_data)
    report_plot(input_data, target, response, mlp.sse_per_iteration, t_label='Sinusoidal function', nLayers=nLayers,
                arch=arch, learning_rate=0.02)


def q2_quadratic():
    nLayers = 2
    arch = [3, 1]
    input_data = [x for x in np.linspace(-2, 2, 100)]
    target = [x ** 2 for x in input_data]
    mlp = MultiLayerPerceptron(nLayers, arch, batch_size=10, lr=0.02, epochs=200000, sse_threshold=0.01,
                               a_function="tanh")
    mlp.SGD(input_data, target)
    response = mlp.feedforward(input_data)
    report_plot(input_data, target, response, mlp.sse_per_iteration, t_label='Quadratic function', nLayers=nLayers,
                arch=arch, learning_rate=0.02)


def q2():
    q2_sin()
    q2_quadratic()
    q2_exp()
    q2_comp()


def q3():
    lr_list = [0.01, 0.03, 0.05, 0.07, 0.09]
    nLayers = 2
    arch = [7, 1]
    input_data = [x for x in np.linspace(-2 * math.pi, 2 * math.pi, 100)]
    target = np.array(np.sin(input_data)).tolist()
    for lr in lr_list:
        mlp = MultiLayerPerceptron(nLayers, arch, batch_size=10, lr=lr, epochs=200000, sse_threshold=0.01,
                                   a_function="tanh")
        mlp.SGD(input_data, target)
        response = mlp.feedforward(input_data)
        report_plot(input_data, target, response, mlp.sse_per_iteration, t_label='Sinusoidal function', nLayers=nLayers,
                    arch=arch, learning_rate=lr)
        print(f"Learning rate: {lr}, SSE: {mlp.sse_per_iteration[-1]}")


def mychirp(t, f0, t1, f1, phase):
    t0 = t[0]
    T = t1 - t0
    k = (f1 - f0) / T
    x = np.sin(2 * np.pi * ((f0 + k / 2 * t) * t + phase))
    return x


def q4():
    f = 100
    step = 1 / f
    t0 = 0
    t1 = 1
    x = np.linspace(t0, t1, 200)
    f0 = 1
    f1 = f / 20
    t = mychirp(x, f0, t1, f1, phase=0)
    plt.plot(x, t)
    plt.show()
    nLayers = 3
    arch = [10, 5, 1]
    input_data = [x for x in np.linspace(t0, t1, 200)]
    target = t
    mlp = MultiLayerPerceptron(nLayers, arch, batch_size=200, lr=0.02, epochs=200000, sse_threshold=0.01,
                               a_function="tanh")
    mlp.SGD(input_data, target)
    response = mlp.feedforward(input_data)
    report_plot(input_data, target, response, mlp.sse_per_iteration, t_label='Chirp function', nLayers=nLayers,
                arch=arch, learning_rate=0.02)


def q5():
    q5_sin()
    q5_quadratic()
    q5_exp()
    q5_comp()


def q5_sin():
    nLayers = 2
    arch = [7, 1]
    input_data = [x for x in np.linspace(-2 * math.pi, 2 * math.pi, 100)]
    target = np.array(np.sin(input_data)).tolist()
    mlp = MultiLayerPerceptron(nLayers, arch, batch_size=100, lr=0.02, epochs=200000, sse_threshold=0.01,
                               a_function="tanh", momentum=0.5)
    mlp.SGD(input_data, target)
    response = mlp.feedforward(input_data)
    report_plot_with_momentum(input_data, target, response, mlp.sse_per_iteration, t_label='Sinusoidal function', nLayers=nLayers,
                arch=arch, learning_rate=0.02, momentum=0.5)


def q5_quadratic():
    nLayers = 2
    arch = [3, 1]
    input_data = [x for x in np.linspace(-2, 2, 100)]
    target = [x ** 2 for x in input_data]
    mlp = MultiLayerPerceptron(nLayers, arch, batch_size=100, lr=0.02, epochs=200000, sse_threshold=0.01,
                               a_function="tanh", momentum=0.5)
    mlp.SGD(input_data, target)
    response = mlp.feedforward(input_data)
    report_plot_with_momentum(input_data, target, response, mlp.sse_per_iteration, t_label='Quadratic function', nLayers=nLayers,
                arch=arch, learning_rate=0.02, momentum=0.5)


def q5_exp():
    nLayers = 2
    arch = [3, 1]
    input_data = [x for x in np.linspace(0, 2, 100)]
    target = [math.exp(x) for x in input_data]
    mlp = MultiLayerPerceptron(nLayers, arch, batch_size=100, lr=0.02, epochs=200000, sse_threshold=0.01,
                               a_function="tanh", momentum=0.5)
    mlp.SGD(input_data, target)
    response = mlp.feedforward(input_data)
    report_plot_with_momentum(input_data, target, response, mlp.sse_per_iteration, t_label='Exponential function', nLayers=nLayers,
                arch=arch, learning_rate=0.02, momentum=0.5)


def q5_comp():
    nLayers = 3
    arch = [10, 5, 1]
    input_data = [x for x in np.linspace(-2 * math.pi, 2 * math.pi, 100)]
    target = [math.sin(x) ** 2 + math.cos(x) ** 3 for x in input_data]
    mlp = MultiLayerPerceptron(nLayers, arch, batch_size=10, lr=0.02, epochs=200000, sse_threshold=0.01,
                               a_function="tanh", momentum=0.5)
    mlp.SGD(input_data, target)
    response = mlp.feedforward(input_data)
    report_plot_with_momentum(input_data, target, response, mlp.sse_per_iteration, t_label='Sinusoidal function', nLayers=nLayers,
                arch=arch, learning_rate=0.02, momentum=0.5)


if __name__ == '__main__':
    # q1()
    # q2()
    # q3()
    # q4()
    # q5()
    # q2_sin()
    # q5_comp()

    np.random.seed(5806)
    q2_sin()
    # np.random.seed(5806)
    # q2_quadratic()