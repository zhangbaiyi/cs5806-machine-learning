import time

import numpy as np
import matplotlib.pyplot as plt


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
        np.random.seed(5806)
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
        self.alist = []
        self.nlist = []
        self.slist = []

    def hail_jafari_feedforward(self, p):
        n = None
        a = None
        self.nlist = []
        tmp = np.array(p).reshape(1, 1)
        a = tmp.copy()
        self.alist = [a]
        for i in range(len(self.weights) - 1):
            n = (self.weights[i] @ a) + self.biases[i]
            self.nlist.append(n)
            a = activate(n, type=self.a_function)
            self.alist.append(a)
        n = (self.weights[-1] @ a) + self.biases[-1]
        self.nlist.append(n)
        a = purelin(n)
        self.alist.append(a)

    def hail_jafari_backprop(self, input_data, target):
        E = [1000]
        iterations = 0
        SSE = 0
        for k in range(len(self.weights)):
            self.slist.append(np.zeros(self.biases[k].shape))
        while E[-1] > self.sse_threshold:
            iterations += 1
            if iterations > self.epochs:
                break
            for p, t in zip(input_data, target):
                self.hail_jafari_feedforward(p)
                # print("Error: ", t - self.alist[-1])
                s_M = -2 * purelin_prime(self.nlist[-1]) * (t - self.alist[-1])
                self.slist[-1] = s_M
                deltas_w = [np.array(self.slist[-1] @ self.alist[-2].T)]
                deltas_b = [np.array(self.slist[-1])]
                for i in range(len(self.weights) - 2, -1, -1):
                    s = np.dot(np.diag(activate_prime(self.nlist[i], type=self.a_function).flatten()),
                               (self.weights[i + 1].T @ self.slist[i + 1]))
                    self.slist[i] = s
                    delta_w = np.array(self.slist[i] @ self.alist[i].T)
                    delta_b = np.array(self.slist[i])
                    dw = [delta_w] + deltas_w
                    db = [delta_b] + deltas_b
                    deltas_w = dw
                    deltas_b = db
                for j in range(len(self.weights)):
                    self.weights[j] = self.weights[j] - (self.lr * deltas_w[j])
                    self.biases[j] = self.biases[j] - (self.lr * deltas_b[j])
                self.hail_jafari_feedforward(p)
                # print("Error after: ", t - self.alist[-1])
                SSE += (t - self.alist[-1]) ** 2
                # print("After: ", p, t)
            E.append(float(SSE))
            SSE = 0
            print(iterations)
            print(E[-1])
        self.sse_per_iteration = E[1:]

    def hail_jafari_backprop_momentum(self, input_data, target):
        E = [1000]
        iterations = 0
        SSE = 0
        for k in range(len(self.weights)):
            self.slist.append(np.zeros(self.biases[k].shape))
        while E[-1] > self.sse_threshold:
            iterations += 1
            if iterations > self.epochs:
                break
            for p, t in zip(input_data, target):
                self.hail_jafari_feedforward(p)
                s_M = -2 * purelin_prime(self.nlist[-1]) * (t - self.alist[-1])
                self.slist[-1] = s_M
                deltas_w = [np.array(self.slist[-1] @ self.alist[-2].T)]
                deltas_b = [np.array(self.slist[-1])]
                for i in range(len(self.weights) - 2, -1, -1):
                    s = np.dot(np.diag(activate_prime(self.nlist[i], type=self.a_function).flatten()),
                               (self.weights[i + 1].T @ self.slist[i + 1]))
                    self.slist[i] = s
                    delta_w = np.array(self.slist[i] @ self.alist[i].T)
                    delta_b = np.array(self.slist[i])
                    dw = [delta_w] + deltas_w
                    db = [delta_b] + deltas_b
                    deltas_w = dw
                    deltas_b = db
                for i in range(len(deltas_w)):
                    deltas_w[i] = self.momentum * self.prev_deltas_w[i] + (1 - self.momentum) * deltas_w[i]
                    deltas_b[i] = self.momentum * self.prev_deltas_b[i] + (1 - self.momentum) * deltas_b[i]
                self.prev_deltas_w = deltas_w
                self.prev_deltas_b = deltas_b
                for j in range(len(self.weights)):
                    self.weights[j] = self.weights[j] - (self.lr * deltas_w[j])
                    self.biases[j] = self.biases[j] - (self.lr * deltas_b[j])
                self.hail_jafari_feedforward(p)
                SSE += (t - self.alist[-1]) ** 2
            E.append(float(SSE))
            SSE = 0
            print(iterations)
            print(E[-1])
        self.sse_per_iteration = E[1:]


def activate(x, type="tanh"):
    """
    Activation function. Hyperbolic tangent function in this program
    :param x: z
    :param type: type of activation function
    :return: activated z
    """
    return np.array(1.0 / (1 + np.exp(-x)))


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
    s = activate(x)
    return s * (1 - s)


def parse_input(input_str):
    """
    Parses the input string to handle 'pi' and numeric values.
    Returns the numeric value of the input.
    """
    if 'pi' in input_str:
        try:
            # Replace 'pi' with 'math.pi' and evaluate the expression
            input_str = input_str.replace('pi', 'np.pi')
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

    target = []
    if a is None or b is None:
        raise ValueError("Invalid input. Please make sure to enter valid numbers.")
    if f == 1:
        input_data = np.linspace(a, b, input_obs)
        target = [[np.sin(x)] for x in np.linspace(a, b, input_obs)]
    elif f == 2:
        input_data = np.linspace(a, b, input_obs)
        target = [[x ** 2] for x in np.linspace(a, b, input_obs)]
    elif f == 3:
        input_data = np.linspace(a, b, input_obs)
        target = [[np.exp(x)] for x in np.linspace(a, b, input_obs)]
    elif f == 4:
        input_data = np.linspace(a, b, input_obs)
        target = [[np.sin(x) ** 2 + np.cos(x) ** 3] for x in np.linspace(a, b, input_obs)]
    else:
        raise ValueError("Invalid input function. Please make sure to enter a valid function.")
    return nLayers, arch, input_data, target


def report_plot_with_momentum(p, target, network_output, sse_per_iteration, t_label="Target", nLayers=0, arch=[],
                              learning_rate=0.01, momentum=0.0):
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
    axs[1].loglog(iterations, sse_per_iteration, linestyle='-')
    axs[1].set_title("SSE: " + str(sse_per_iteration[-1]) + "\n" + "Learning rate: " + str(
        learning_rate) + "\n" + "Number of iterations: " + str(len(sse_per_iteration)) + "\n" + "Architecture: " + str(
        arch) + "\n" + "SSE threshold: " + str(0.01) + "\n" + "Momentum: " + str(momentum))
    axs[1].set_xlabel('Number of iterations (log scale)')
    axs[1].set_ylabel('Magnitude (log scale)')
    axs[1].grid()
    axs[1].legend(['SSE'])
    plt.tight_layout()
    plt.savefig(t_label + "_with_m_" + time.strftime("%Y%m%d-%H%M%S") + '.png')
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
    axs[1].loglog(iterations, sse_per_iteration, linestyle='-')
    axs[1].set_title("SSE: " + str(sse_per_iteration[-1]) + "\n" + "Learning rate: " + str(
        learning_rate) + "\n" + "Number of iterations: " + str(len(sse_per_iteration)) + "\n" + "Architecture: " + str(
        arch) + "\n" + "SSE threshold: " + str(0.01))
    axs[1].set_xlabel('Number of iterations (log scale)')
    axs[1].set_ylabel('Magnitude (log scale)')
    axs[1].grid()
    axs[1].legend(['SSE'])

    plt.tight_layout()
    plt.savefig(t_label + time.strftime("%Y%m%d-%H%M%S") + '.png')
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
    input_data = [x for x in np.linspace(-2 * np.pi, 2 * np.pi, 100)]
    mlp = MultiLayerPerceptron(nLayers, arch, batch_size=100, lr=0.01, epochs=10000, sse_threshold=0.02,
                               a_function="sigmoid")
    target = [np.sin(x) for x in input_data]
    network_output = []
    for i in range(len(input_data)):
        mlp.hail_jafari_feedforward(input_data[i])
        network_output.append(mlp.alist[-1][0][0])
    q1_plt_target_and_response(input_data, target, network_output)
    print("Weights and biases initialized:")
    for i in range(len(mlp.weights)):
        print(f"Layer {i + 1} weights (dim: {mlp.weights[i].shape}):\n {mlp.weights[i]}")
        print(f"Layer {i + 1} biases (dim: {mlp.biases[i].shape}):\n {mlp.biases[i]}")


def q2_comp():
    nLayers = 3
    arch = [10, 5, 1]
    input_data = [x for x in np.linspace(-2 * np.pi, 2 * np.pi, 100)]
    target = [np.sin(x) ** 2 + np.cos(x) ** 3 for x in input_data]
    mlp = MultiLayerPerceptron(nLayers, arch, batch_size=100, lr=0.02, epochs=200000, sse_threshold=0.01,
                               a_function="sigmoid")
    print("Initial weights and biases:")
    print(mlp.weights)
    print(mlp.biases)
    mlp.hail_jafari_backprop(input_data, target)
    network_output = []
    for i in range(len(input_data)):
        mlp.hail_jafari_feedforward(input_data[i])
        network_output.append(mlp.alist[-1][0][0])
    report_plot(input_data, target, network_output, mlp.sse_per_iteration, t_label='Complex function',
                nLayers=nLayers,
                arch=arch, learning_rate=0.02)
    print("Final weights and biases:")
    print(mlp.weights)
    print(mlp.biases)


def q2_exp():
    nLayers = 2
    arch = [3, 1]
    input_data = [x for x in np.linspace(0, 2, 100)]
    target = [np.exp(x) for x in input_data]
    mlp = MultiLayerPerceptron(nLayers, arch, batch_size=100, lr=0.02, epochs=200000, sse_threshold=0.01,
                               a_function="sigmoid")
    print("Initial weights and biases:")
    print(mlp.weights)
    print(mlp.biases)
    mlp.hail_jafari_backprop(input_data, target)
    network_output = []
    for i in range(len(input_data)):
        mlp.hail_jafari_feedforward(input_data[i])
        network_output.append(mlp.alist[-1][0][0])
    report_plot(input_data, target, network_output, mlp.sse_per_iteration, t_label='Exponential function',
                nLayers=nLayers,
                arch=arch, learning_rate=0.02)
    print("Final weights and biases:")
    print(mlp.weights)
    print(mlp.biases)


def q2_sin():
    nLayers = 2
    arch = [7, 1]
    input_data = [x for x in np.linspace(-2 * np.pi, 2 * np.pi, 100)]

    target = np.array(np.sin(input_data)).tolist()
    mlp = MultiLayerPerceptron(nLayers, arch, batch_size=100, lr=0.02, epochs=200000, sse_threshold=0.01,
                               a_function="sigmoid")
    print("Initial weights and biases:")
    print(mlp.weights)
    print(mlp.biases)
    mlp.hail_jafari_backprop(input_data, target)
    network_output = []
    for i in range(len(input_data)):
        mlp.hail_jafari_feedforward(input_data[i])
        network_output.append(mlp.alist[-1][0][0])
    report_plot(input_data, target, network_output, mlp.sse_per_iteration, t_label='Sinusoidal function',
                nLayers=nLayers,
                arch=arch, learning_rate=0.02)
    print("Final weights and biases:")
    print(mlp.weights)
    print(mlp.biases)


def q2_quadratic():
    nLayers = 2
    arch = [3, 1]
    input_data = [x for x in np.linspace(-2, 2, 100)]

    target = [x ** 2 for x in input_data]
    mlp = MultiLayerPerceptron(nLayers, arch, batch_size=100, lr=0.02, epochs=200000, sse_threshold=0.01,
                               a_function="sigmoid")
    print("Initial weights and biases:")
    print(mlp.weights)
    print(mlp.biases)
    mlp.hail_jafari_backprop(input_data, target)
    network_output = []
    for i in range(len(input_data)):
        mlp.hail_jafari_feedforward(input_data[i])
        network_output.append(mlp.alist[-1][0][0])
    report_plot(input_data, target, network_output, mlp.sse_per_iteration, t_label='Quadratic function',
                nLayers=nLayers,
                arch=arch, learning_rate=0.02)
    print("Final weights and biases:")
    print(mlp.weights)
    print(mlp.biases)


def q3():
    lr_list = [0.01, 0.03, 0.05, 0.07, 0.09]
    nLayers = 2
    arch = [7, 1]
    input_data = [x for x in np.linspace(-2 * np.pi, 2 * np.pi, 100)]
    target = np.array(np.sin(input_data)).tolist()
    for lr in lr_list:
        print("=====================================")
        print("Learning rate: ", lr)
        mlp = MultiLayerPerceptron(nLayers, arch, batch_size=100, lr=lr, epochs=200000, sse_threshold=0.01,
                                   a_function="sigmoid")
        print("Initial weights and biases:")
        print(mlp.weights)
        print(mlp.biases)
        mlp.hail_jafari_backprop(input_data, target)
        network_output = []
        for i in range(len(input_data)):
            mlp.hail_jafari_feedforward(input_data[i])
            network_output.append(mlp.alist[-1][0][0])
        report_plot(input_data, target, network_output, mlp.sse_per_iteration, t_label='Sinusoidal function',
                    nLayers=nLayers,
                    arch=arch, learning_rate=lr)
        print("Final weights and biases:")
        print(mlp.weights)
        print(mlp.biases)


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
                               a_function="sigmoid")
    print("Initial weights and biases:")
    print(mlp.weights)
    print(mlp.biases)
    mlp.hail_jafari_backprop(input_data, target)
    network_output = []
    for i in range(len(input_data)):
        mlp.hail_jafari_feedforward(input_data[i])
        network_output.append(mlp.alist[-1][0][0])
    report_plot(input_data, target, network_output, mlp.sse_per_iteration, t_label='chirp',
                nLayers=nLayers,
                arch=arch, learning_rate=mlp.lr)
    print("Final weights and biases:")
    print(mlp.weights)
    print(mlp.biases)


def q5_sin():
    nLayers = 2
    arch = [7, 1]
    input_data = [x for x in np.linspace(-2 * np.pi, 2 * np.pi, 100)]
    target = np.array(np.sin(input_data)).tolist()
    mlp = MultiLayerPerceptron(nLayers, arch, batch_size=100, lr=0.02, epochs=200000, sse_threshold=0.01,
                               a_function="sigmoid", momentum=0.5)
    mlp.hail_jafari_backprop_momentum(input_data, target)
    network_output = []
    for i in range(len(input_data)):
        mlp.hail_jafari_feedforward(input_data[i])
        network_output.append(mlp.alist[-1][0][0])
    report_plot_with_momentum(input_data, target, network_output, mlp.sse_per_iteration, t_label='Sinusoidal function',
                nLayers=nLayers,
                arch=arch, learning_rate=mlp.lr, momentum=mlp.momentum)
    print("Final weights and biases:")
    print(mlp.weights)
    print(mlp.biases)


def q5_quadratic():
    nLayers = 2
    arch = [3, 1]
    input_data = [x for x in np.linspace(-2, 2, 100)]
    target = [x ** 2 for x in input_data]
    mlp = MultiLayerPerceptron(nLayers, arch, batch_size=100, lr=0.02, epochs=200000, sse_threshold=0.01,
                               a_function="sigmoid", momentum=0.5)
    mlp.hail_jafari_backprop_momentum(input_data, target)
    network_output = []
    for i in range(len(input_data)):
        mlp.hail_jafari_feedforward(input_data[i])
        network_output.append(mlp.alist[-1][0][0])
    report_plot_with_momentum(input_data, target, network_output, mlp.sse_per_iteration, t_label='Quadratic function',
                nLayers=nLayers,
                arch=arch, learning_rate=mlp.lr, momentum=mlp.momentum)
    print("Final weights and biases:")
    print(mlp.weights)
    print(mlp.biases)


def q5_exp():
    nLayers = 2
    arch = [3, 1]
    input_data = [x for x in np.linspace(0, 2, 100)]
    target = [np.exp(x) for x in input_data]
    mlp = MultiLayerPerceptron(nLayers, arch, batch_size=100, lr=0.02, epochs=200000, sse_threshold=0.01,
                               a_function="sigmoid", momentum=0.5)
    mlp.hail_jafari_backprop_momentum(input_data, target)
    network_output = []
    for i in range(len(input_data)):
        mlp.hail_jafari_feedforward(input_data[i])
        network_output.append(mlp.alist[-1][0][0])
    report_plot_with_momentum(input_data, target, network_output, mlp.sse_per_iteration, t_label='Exponential function',
                nLayers=nLayers,
                arch=arch, learning_rate=mlp.lr, momentum=mlp.momentum)
    print("Final weights and biases:")
    print(mlp.weights)
    print(mlp.biases)


def q5_comp():
    nLayers = 3
    arch = [10, 5, 1]
    input_data = [x for x in np.linspace(-2 * np.pi, 2 * np.pi, 100)]
    target = [np.sin(x) ** 2 + np.cos(x) ** 3 for x in input_data]
    mlp = MultiLayerPerceptron(nLayers, arch, batch_size=100, lr=0.02, epochs=200000, sse_threshold=0.01,
                               a_function="sigmoid", momentum=0.5)
    mlp.hail_jafari_backprop_momentum(input_data, target)
    network_output = []
    for i in range(len(input_data)):
        mlp.hail_jafari_feedforward(input_data[i])
        network_output.append(mlp.alist[-1][0][0])
    report_plot_with_momentum(input_data, target, network_output, mlp.sse_per_iteration, t_label='Complex function',
                              nLayers=nLayers,
                              arch=arch, learning_rate=mlp.lr, momentum=mlp.momentum)
    print("Final weights and biases:")
    print(mlp.weights)
    print(mlp.biases)


def q6():
    momentum = [0.2, 0.4, 0.6, 0.8, 0.9]
    nLayers = 2
    arch = [7, 1]
    input_data = [x for x in np.linspace(-2 * np.pi, 2 * np.pi, 100)]
    target = np.array(np.sin(input_data)).tolist()
    for m in momentum:
        print("=====================================")
        print("Momentum: ", m)
        mlp = MultiLayerPerceptron(nLayers, arch, batch_size=100, lr=0.02, epochs=200000, sse_threshold=0.01,
                                   a_function="sigmoid", momentum=m)
        print("Initial weights and biases:")
        print(mlp.weights)
        print(mlp.biases)
        mlp.hail_jafari_backprop_momentum(input_data, target)
        network_output = []
        for i in range(len(input_data)):
            mlp.hail_jafari_feedforward(input_data[i])
            network_output.append(mlp.alist[-1][0][0])
        report_plot_with_momentum(input_data, target, network_output, mlp.sse_per_iteration,
                                  t_label='Sinusoidal function',
                                  nLayers=nLayers,
                                  arch=arch, learning_rate=0.02, momentum=m)
        print("Final weights and biases:")
        print(mlp.weights)
        print(mlp.biases)


def prompt_and_plot():
    nLayers, arch, input_data, target = mlp_init_prompt()
    mlp = MultiLayerPerceptron(nLayers, arch, batch_size=100, lr=0.01, epochs=10000, sse_threshold=0.02,
                               a_function="sigmoid")
    print("Hello")
    print("Initial weights and biases:")
    print(mlp.weights)
    print(mlp.biases)
    mlp.hail_jafari_backprop(input_data, target)
    network_output = []
    for i in range(len(input_data)):
        mlp.hail_jafari_feedforward(input_data[i])
        network_output.append(mlp.alist[-1][0][0])
    report_plot(input_data, target, network_output, mlp.sse_per_iteration, t_label='User-defined function',
                nLayers=nLayers,
                arch=arch, learning_rate=0.01)


if __name__ == '__main__':
    q1()
    q2_sin()
    q2_quadratic()
    q2_exp()
    q2_comp()
    q3()
    q4()
    q5_sin()
    q5_quadratic()
    q5_exp()
    q5_comp()
    q6()
    prompt_and_plot()

