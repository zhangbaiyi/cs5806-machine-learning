#
# Please make sure to do the following to the get the correct response:
#
# Set the np random seed to 5806.
# Use np.random.randn() for all weights in both layers.
# Use the random.rand() for bias in the first layer.
# Use the random.randn() for the bias in the second layer.
import time

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(5806)


# Q1 - Paper based
#
# Q2 - Simulate RBF network - 2 layers
# Parameters:
w_11_1 = -1
w_21_1 = 1
b_1_1 = 2
b_2_1 = 2
w_11_2 = 1
w_12_2 = 1
b_1_2 = 0
w_1 = np.array([w_11_1, w_21_1]).reshape(2, 1)
w_2 = np.array([w_11_2, w_12_2]).reshape(1, 2)
p_dummy = np.array([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])

# Layer 1 - Radial Basis Layer
# a_i_1 = radbas(||w_i_1 - p|| * b_i_1)
# radbas = exp(-x^2)

a_1 = []
for p in p_dummy:
    a_1_1 = np.exp(-np.square(np.abs(w_11_1 - p) * b_1_1))
    a_2_1 = np.exp(-np.square(np.abs(w_21_1 - p) * b_2_1))
    a_1.append([a_1_1, a_2_1])
print(a_1)

# Layer 2 - Linear Layer
a_2 = []
for a in a_1:
    a_1_2 = w_2 @ np.array(a).reshape(2, 1) + b_1_2
    a_2.append(a_1_2[0][0])

print(a_2)

# Q2 - 1: Plot
plt.plot(p_dummy, a_2, label='RBF response without training', color='blue', linewidth=2)
plt.title('RBF network with 2 neurons in the hidden layer - 9 obs')
plt.xlabel('p')
plt.ylabel('Mag.')
plt.legend()
plt.grid(True)
plt.show()

# Q2 - Continuous p
p_dense = np.linspace(-2, 2, 1000)
a_1 = []
for p in p_dense:
    a_1_1 = np.exp(-np.square(np.abs(w_11_1 - p) * b_1_1))
    a_2_1 = np.exp(-np.square(np.abs(w_21_1 - p) * b_2_1))
    a_1.append([a_1_1, a_2_1])

a_2 = []
for a in a_1:
    a_1_2 = w_2 @ np.array(a).reshape(2, 1) + b_1_2
    a_2.append(a_1_2[0][0])

plt.plot(p_dense, a_2, label='RBF response without training', color='blue', linewidth=2)
plt.title('RBF network with 2 neurons in the hidden layer - 1000 obs')
plt.xlabel('p')
plt.ylabel('Mag.')
plt.legend()
plt.grid(True)
plt.show()

# Q3
# 3- Using python program develop a code that simulates the RBF network in the general case. A code should
# be written in the way that a user enters the following information.
# • Weight initialization: Enter the mean of the normally distributed weights =
# • Weight initialization: Enter the variance of normally distributed weights =
# • Enter the lower bound for the input p =
# • Enter the upper bound for the input p =
# • Enter the number of samples =
# • Enter the number of neurons in the hidden layer =
# • Enter the number of neurons in the output layer =
def q3_prompt():
    mean = float(input('Weight initialization: Enter the mean of the normally distributed weights = '))
    variance = float(input('Weight initialization: Enter the variance of normally distributed weights = '))
    lower = float(input('Enter the lower bound for the input p = '))
    upper = float(input('Enter the upper bound for the input p = '))
    samples = int(input('Enter the number of samples = '))
    hidden = int(input('Enter the number of neurons in the hidden layer = '))
    output = int(input('Enter the number of neurons in the output layer = '))
    return mean, variance, lower, upper, samples, hidden, output

def q3():
    np.random.seed(5806)
    # mean, variance, lower, upper, samples, hidden, output = q3_prompt()
    mean = 0
    variance = 1
    lower = -2
    upper = 2
    samples = 100
    hidden = 10
    output = 1
    w_1 = mean + np.sqrt(variance) * np.random.randn(hidden)
    b_1 = mean + np.sqrt(variance) * np.random.rand(hidden)
    w_2 = mean + np.sqrt(variance) * np.random.randn(output, hidden)
    b_2 = mean + np.sqrt(variance) * np.random.randn(output)



    p_dummy = np.linspace(lower, upper, samples)
    a_1 = []
    for p in p_dummy:
        a = []
        for i in range(hidden):
            a_i = np.exp(-np.square(np.abs(w_1[i] - p) * b_1[i]))
            a.append(a_i)
        a_1.append(a)
    a_2 = []
    for a in a_1:
        a_1_2 = w_2 @ np.array(a).reshape(hidden, 1) + b_2
        a_2.append(a_1_2[0][0])
    plt.plot(p_dummy, a_2, label='RBF response without training', color='blue', linewidth=2)
    plt.title(f'RBF network with {hidden} neurons in the hidden layer - {samples} obs')
    plt.xlabel('p')
    plt.ylabel('Mag.')
    plt.legend()
    plt.grid(True)
    plt.savefig(time.strftime("%Y%m%d-%H%M%S") + 'q3.png')
    plt.show()

q3()


# illustrates the effects of parameter changes on
# the network response. The blue curve is the nominal
# response. The other curves correspond to the network
# response when one parameter at a time is varied over
# the following ranges:
w_21_1_range = np.linspace(0, 2, 5)
w_11_2_range = np.linspace(-1, 1, 5)
b_2_1_range = np.linspace(0.5, 8, 5)
b_2_range = np.linspace(-1, 1, 5)
# Fixed values
# 𝑤11_1=−1, 𝑤21_1 = 1, 𝑏1_1 = 2, 𝑏2_1 = 2, 𝑤11_2 = 1, 𝑤12_2 = 1, 𝑏1 2 = 0
w_11_1 = -1
w_21_1 = 1
w_1 = np.array([w_11_1, w_21_1]).reshape(2, 1)
b_1_1 = 2
b_2_1 = 2
b_1 = np.array([b_1_1, b_2_1]).reshape(2, 1)
w_11_2 = 1
w_12_2 = 1
w_2 = np.array([w_11_2, w_12_2]).reshape(1, 2)
b_1_2 = 0

p_dummy_q4 = np.linspace(-2, 2, 1000)

def range_plot_iterate_over_b_2_1():
    # coolwarm colors
    colors = plt.cm.coolwarm(np.linspace(0, 1, 5))
    for b_2_1_i, color in zip(b_2_1_range, colors):
        a_1 = []
        for p in p_dummy_q4:
            ps = np.array([p, p]).reshape(2, 1)
            a_1_1 = np.exp(-np.square(np.abs(w_11_1 - p) * b_1_1))
            a_2_1 = np.exp(-np.square(np.abs(w_21_1 - p) * b_2_1_i))
            a_1.append([a_1_1, a_2_1])
        a_2 = []
        for a in a_1:
            a_1_2 = w_2 @ np.array(a).reshape(2, 1) + b_1_2
            a_2.append(a_1_2[0][0])
        plt.plot(p_dummy_q4, a_2, label=f'b_2_1 = {b_2_1_i}', linewidth=2, color=color)
    plt.title('RBF network with 2 neurons in the hidden layer - 1000 obs')
    plt.xlabel('p')
    plt.ylabel('Mag.')
    plt.legend()
    plt.grid(True)
    plt.savefig(time.strftime("%Y%m%d-%H%M%S") + 'b_2_1.png')  # 'b_2_1.png'
    plt.show()


def range_plot_iterate_over_w_21_1():
    # coolwarm colors
    colors = plt.cm.coolwarm(np.linspace(0, 1, 5))
    for w_21_1_i, color in zip(w_21_1_range, colors):
        a_1 = []
        for p in p_dummy_q4:
            a_1_1 = np.exp(-np.square(np.abs(w_11_1 - p) * b_1_1))
            a_2_1 = np.exp(-np.square(np.abs(w_21_1_i - p) * b_2_1))
            a_1.append([a_1_1, a_2_1])
        a_2 = []
        for a in a_1:
            a_1_2 = w_2 @ np.array(a).reshape(2, 1) + b_1_2
            a_2.append(a_1_2[0][0])
        plt.plot(p_dummy_q4, a_2, label=f'w_21_1 = {w_21_1_i}', linewidth=2, color=color)
    plt.title('RBF network with 2 neurons in the hidden layer - 1000 obs')
    plt.xlabel('p')
    plt.ylabel('Mag.')
    plt.legend()
    plt.grid(True)
    plt.savefig(time.strftime("%Y%m%d-%H%M%S") + 'w_21_1.png')
    plt.show()


def range_plot_iterate_over_w_11_2():
    # coolwarm colors
    colors = plt.cm.coolwarm(np.linspace(0, 1, 5))
    for w_11_2_i, color in zip(w_11_2_range, colors):
        a_1 = []
        for p in p_dummy_q4:
            ps = np.array([p, p]).reshape(2, 1)
            a_1_1 = np.exp(-np.square(np.abs(w_11_1 - p) * b_1_1))
            a_2_1 = np.exp(-np.square(np.abs(w_21_1 - p) * b_2_1))
            a_1.append([a_1_1, a_2_1])
        a_2 = []
        for a in a_1:
            a_1_2 = np.array([w_11_2_i, w_12_2]) @ np.array(a).reshape(2, 1) + b_1_2
            a_2.append(a_1_2[0])
        plt.plot(p_dummy_q4, a_2, label=f'w_11_2 = {w_11_2_i}', linewidth=2, color=color)
    plt.title('RBF network with 2 neurons in the hidden layer - 1000 obs')
    plt.xlabel('p')
    plt.ylabel('Mag.')
    plt.legend()
    plt.grid(True)
    plt.savefig(time.strftime("%Y%m%d-%H%M%S") + 'w_11_2.png')
    plt.show()


def range_plot_iterate_over_b_2():
    # coolwarm colors
    colors = plt.cm.coolwarm(np.linspace(0, 1, 5))
    for b_2_i, color in zip(b_2_range, colors):
        a_1 = []
        for p in p_dummy_q4:
            ps = np.array([p, p]).reshape(2, 1)
            a_1_1 = np.exp(-np.square(np.abs(w_11_1 - p) * b_1_1))
            a_2_1 = np.exp(-np.square(np.abs(w_21_1 - p) * b_2_1))
            a_1.append([a_1_1, a_2_1])
        a_2 = []
        for a in a_1:
            a_1_2 = w_2 @ np.array(a).reshape(2, 1) + b_2_i
            a_2.append(a_1_2[0][0])
        plt.plot(p_dummy_q4, a_2, label=f'b_2 = {b_2_i}', linewidth=2, color=color)
    plt.title('RBF network with 2 neurons in the hidden layer - 1000 obs')
    plt.xlabel('p')
    plt.ylabel('Mag.')
    plt.legend()
    plt.grid(True)
    plt.savefig(time.strftime("%Y%m%d-%H%M%S") + 'b_2.png')
    plt.show()

range_plot_iterate_over_b_2_1()
range_plot_iterate_over_w_21_1()
range_plot_iterate_over_w_11_2()
range_plot_iterate_over_b_2()

# Q4
class RBF:
    # 2 hidden neurons
    # 2 layered network with one output
    def __init__(self, epochs, sse_threshold, lr, momentum=0.0):
        np.random.seed(5806)
        weights = []
        biases = []
        # Network: 1st layer: 2x1, 2nd layer: 1x2, output: 1x1
        # Init: mean = 0, variance = 1

        layer_sizes = [1, 2, 1]
        for i in range(len(layer_sizes) - 1):
            weights.append(
                np.random.randn(layer_sizes[i + 1], layer_sizes[i]).reshape(layer_sizes[i + 1], layer_sizes[i]))
            if i == 0:
                biases.append(np.random.rand(layer_sizes[1]).reshape(layer_sizes[1], 1))
            else:
                biases.append(np.random.randn(layer_sizes[2]).reshape(layer_sizes[2], 1))

        # layer_sizes = [1, 2, 1]
        # weights.append(np.array([[0], [0]]))
        # weights.append(np.array([[-2, -2]]))
        # biases.append(np.array([[1], [1]]))
        # biases.append(np.array([[1]]))

        self.weights = weights
        self.biases = biases
        self.alist = []
        self.nlist = []
        self.slist = []
        self.epochs = epochs
        self.sse_threshold = sse_threshold
        self.lr = lr
        self.momentum = momentum
        self.sse_per_iteration = []
        self.prev_deltas_w = [np.zeros_like(w) for w in self.weights]
        self.prev_deltas_b = [np.zeros_like(b) for b in self.biases]

    def plot(self, x, y):
        actual = []
        for p in x:
            self.forward(p)
            actual.append(self.alist[-1][0][0])
        # plt.plot(x, actual, 'g-', label='RBF response with training')
        # plt.plot(x, y, 'ro', markersize=4, label='Target')
        # plt.title('RBF network with 2 neurons in the hidden layer - 1000 obs')
        # plt.xlabel('p')
        # plt.ylabel('Mag.')
        # plt.legend()
        # plt.grid(True)
        # plt.show()

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].plot(x, y, 'ro', label="Target")
        axs[0].plot(x, actual, 'g-', label='Network output')
        axs[0].set_title("Number of observations: " + str(len(x)) +
                         "\n" + "Number of iterations: " + str(len(self.sse_per_iteration)) +
                         "\n" + "Learning rate: " + str(self.lr) +
                         "\n" + "Momentum term: " + str(self.momentum) +
                         "\n" + "Number of neurons: " + str(2))
        axs[0].set_xlabel('p')
        axs[0].set_ylabel('Magnitude')
        axs[0].grid()
        axs[0].legend()

        iterations = range(1, len(self.sse_per_iteration) + 1)
        axs[1].loglog(iterations, self.sse_per_iteration, linestyle='-')
        axs[1].set_title("SSE: " + str(self.sse_per_iteration[-1]) +
                         "\n" + "Learning rate: " + str(self.lr) +
                         "\n" + "Number of iterations: " + str(len(self.sse_per_iteration)) +
                         "\n" + "Number of neurons: " + str(2) +
                         "\n" + "SSE cut off: " + str(self.sse_threshold))
        axs[1].set_xlabel('Number of iterations (log scale)')
        axs[1].set_ylabel('Magnitude (log scale)')
        axs[1].grid()
        axs[1].legend(['SSE'])
        plt.tight_layout()
        plt.savefig(
            "lr_" + str(self.lr) + "_gamma_" + str(self.momentum) + "_" + time.strftime("%Y%m%d-%H%M%S") + '.png')
        plt.show()

    def forward(self, p):
        n = None
        a = None
        self.nlist = []
        tmp = np.full_like(self.weights[0], p)
        a = tmp.copy()
        self.alist = [a]
        # First layer radbas(||w - p|| * b)
        n = np.abs(a - self.weights[0]) * (self.biases[0])
        self.nlist.append(n)
        a = radbas(n)
        self.alist.append(a)
        # Second Layer w @ a + b
        n = (self.weights[1] @ a) + self.biases[1]
        self.nlist.append(n)
        a = n
        self.alist.append(a)

    def train_nonlinear_optimization(self, input, target):
        E = [1000]
        iterations = 0
        SSE = 0
        for k in range(2):
            self.slist.append(np.zeros(self.biases[k].shape))
        while E[-1] > self.sse_threshold:
            iterations += 1
            if iterations > self.epochs:
                break
            for p, t in zip(input, target):
                self.forward(p)
                # Error: t-a
                e = t - self.alist[-1][0][0]
                # Layer 2
                s_M = -2 * purelin_prime(self.nlist[1]) * e
                self.slist[1] = s_M
                deltas_w = [np.array(self.slist[1] * self.alist[1].T)]
                deltas_b = [np.array(self.slist[1]).reshape(1, 1)]
                # Layer 1
                s = np.diag(radbas_prime(self.nlist[0]).flatten()) @ (self.weights[1].T * self.slist[1])
                self.slist[0] = s
                p_as_list = np.full_like(self.weights[0], p)
                delta_w = s * self.biases[0].reshape(2, 1) * (self.weights[0] - p_as_list) / np.abs(
                    p_as_list - self.weights[0])
                delta_b = s * np.abs(p_as_list - self.weights[0])
                deltas_w = [delta_w] + deltas_w
                deltas_b = [delta_b] + deltas_b
                for i in range(2):
                    self.weights[i] = self.weights[i] - (self.lr * deltas_w[i])
                    self.biases[i] = self.biases[i] - (self.lr * deltas_b[i])
                self.forward(p)
                SSE += (t - self.alist[-1][0][0]) ** 2
            E.append(float(SSE))
            # print(f'Epoch: {iterations}, SSE: {SSE}')
            # print(self.weights)
            # print(self.biases)
            SSE = 0
        self.sse_per_iteration = E

    def train_nonlinear_optimization_with_momentum(self, input, target):
        E = [1000]
        iterations = 0
        SSE = 0
        for k in range(2):
            self.slist.append(np.zeros(self.biases[k].shape))
        while E[-1] > self.sse_threshold:
            iterations += 1
            if iterations > self.epochs:
                break
            for p, t in zip(input, target):
                self.forward(p)
                # Error: t-a
                e = t - self.alist[-1][0][0]
                # Layer 2
                s_M = -2 * purelin_prime(self.nlist[1]) * e
                self.slist[1] = s_M
                deltas_w = [np.array(self.slist[1] * self.alist[1].T)]
                deltas_b = [np.array(self.slist[1]).reshape(1, 1)]
                # Layer 1
                s = np.diag(radbas_prime(self.nlist[0]).flatten()) @ (self.weights[1].T * self.slist[1])
                self.slist[0] = s
                p_as_list = np.full_like(self.weights[0], p)
                delta_w = s * self.biases[0].reshape(2, 1) * (self.weights[0] - p_as_list) / np.abs(
                    p_as_list - self.weights[0])
                delta_b = s * np.abs(p_as_list - self.weights[0])
                deltas_w = [delta_w] + deltas_w
                deltas_b = [delta_b] + deltas_b
                for i in range(len(deltas_w)):
                    deltas_w[i] = self.momentum * self.prev_deltas_w[i] + (1 - self.momentum) * deltas_w[i]
                    deltas_b[i] = self.momentum * self.prev_deltas_b[i] + (1 - self.momentum) * deltas_b[i]
                self.prev_deltas_w = deltas_w
                self.prev_deltas_b = deltas_b

                for i in range(2):
                    self.weights[i] = self.weights[i] - (self.lr * deltas_w[i])
                    self.biases[i] = self.biases[i] - (self.lr * deltas_b[i])
                self.forward(p)
                SSE += (t - self.alist[-1][0][0]) ** 2
            E.append(float(SSE))
            # print(f'Epoch: {iterations}, SSE: {SSE}')
            # print(self.weights)
            # print(self.biases)
            SSE = 0
        self.sse_per_iteration = E


class RBF_1_Layer:
    # 1 hidden neurons
    # 2 layered network with one output
    def __init__(self, epochs=2000, sse_threshold=0.001, lr=0.01):
        np.random.seed(5806)
        weights = []
        biases = []
        # Network: 1st layer: 2x1, 2nd layer: 1x2, output: 1x1
        # Init: mean = 0, variance = 1
        weights = [np.array([0]).reshape(1, 1), np.array([-2]).reshape(1, 1)]
        b = [np.array([1]).reshape(1, 1), np.array([1]).reshape(1, 1)]
        self.weights = weights
        self.biases = b
        self.alist = []
        self.nlist = []
        self.slist = []
        self.epochs = epochs
        self.sse_threshold = sse_threshold
        self.lr = lr
        self.sse_per_iteration = []

    def forward(self, p):
        n = None
        a = None
        self.nlist = []
        tmp = np.full_like(self.weights[0], p)
        a = tmp.copy()
        self.alist = [a]
        # First layer radbas(||w - p|| * b)
        n = np.linalg.norm(x=a - self.weights[0], ord=2, keepdims=True, axis=1) * (self.biases[0])
        self.nlist.append(n)
        a = radbas(n)
        self.alist.append(a)
        # Second Layer w @ a + b
        n = (self.weights[1] @ a) + self.biases[1]
        self.nlist.append(n)
        a = n
        self.alist.append(a)

    def plot(self, x, y):
        actual = []
        for p in x:
            self.forward(p)
            actual.append(self.alist[-1][0][0])
        plt.plot(x, actual, 'g-', label='RBF response with training')
        plt.plot(x, y, 'ro', markersize=4, label='Target')
        plt.title('RBF network with 2 neurons in the hidden layer - 1000 obs')
        plt.xlabel('p')
        plt.ylabel('Mag.')
        plt.legend()
        plt.grid(True)
        plt.show()

    def train_nonlinear_optimization(self, input, target):
        E = [1000]
        iterations = 0
        SSE = 0
        for k in range(2):
            self.slist.append(np.zeros(self.biases[k].shape))
        while E[-1] > self.sse_threshold:
            iterations += 1
            if iterations > self.epochs:
                break
            for p, t in zip(input, target):
                self.forward(p)
                # Error: t-a
                e = t - self.alist[-1][0][0]
                # Layer 2
                s_M = -2 * purelin_prime(self.nlist[1]) * e
                self.slist[1] = s_M
                deltas_w = [np.array(self.slist[1] * self.alist[1].T)]
                deltas_b = [np.array(self.slist[1])]
                # Layer 1
                s = np.diag(radbas_prime(self.nlist[0]).flatten()) @ ((self.weights[1].T) * self.slist[1])
                self.slist[0] = s
                p_as_list = np.full_like(self.weights[0], p)
                delta_w = s * self.biases[0].reshape(1, 1) * (self.weights[0] - p_as_list) / np.linalg.norm(
                    x=p_as_list - self.weights[0], ord=2, keepdims=True, axis=0)
                delta_b = s * np.linalg.norm(x=p_as_list - self.weights[0], ord=2, keepdims=True, axis=0)
                deltas_w = [delta_w] + deltas_w
                deltas_b = [delta_b] + deltas_b
                for i in range(2):
                    self.weights[i] = self.weights[i] - (self.lr * deltas_w[i])
                    self.biases[i] = self.biases[i] - (self.lr * deltas_b[i])
                self.forward(p)
                SSE += (t - self.alist[-1]) ** 2
            E.append(float(SSE))
            print(f'Epoch: {iterations}, SSE: {SSE}')
            SSE = 0
        self.sse_per_iteration = E


def purelin(x):
    return x


def purelin_prime(x):
    return 1


def radbas(x):
    # a = radbas(n) = exp(-n^2)
    return np.exp(-np.square(x))


def radbas_prime(x):
    # a = radbas(n) = exp(-n^2)
    # a' = -2n * exp(-n^2)
    return -2 * x * np.exp(-np.square(x))


# rbf = RBF(epochs=2000, sse_threshold=0.001, lr=0.02, momentum=0.04)
# x = np.linspace(0, np.pi, 100)
# y = np.sin(x)
# rbf.train_nonlinear_optimization_with_momentum(x, y)
# rbf.plot(x, y)


# rbf = RBF(epochs=2000, sse_threshold=0.001, lr=0.01)
# x = np.linspace(0, np.pi, 100)
# y = np.sin(x)
# rbf.train_nonlinear_optimization(x, y)
# rbf.plot(x, y)

# Repeat question 4 with the momentum term. Fix the learning ratio to 0.01 and vary the gamma value
# as shown in the following table. What is the effect of gamma terms in the training? What value of gamma
# do you select for this function approximation problem? All the settings remain the same as the previous
# question. Fill out the table below. Plot the Network response versus target and the SSE as shown below
# with the information added on the title. Display the initial and final trained weights and biases on the
# console.
#

# rbf = RBF(epochs=1, sse_threshold=0.001, lr=1)
# rbf.train_nonlinear_optimization([-1], [1])

# rbf = RBF_1_Layer(epochs=1, sse_threshold=0.001, lr=1)
# rbf.train_nonlinear_optimization([-1], [1])


def q4():
    lrs = [0.01, 0.02, 0.04, 0.06, 0.08, 0.1]
    for lr in lrs:
        rbf = RBF(epochs=2000, sse_threshold=0.001, lr=lr, momentum=0)
        print("INITIAL WEIGHTS")
        for i in range(len(rbf.weights)):
            rounded_weights = np.round(rbf.weights[i], 2)
            rounded_biases = np.round(rbf.biases[i], 2)
            print(f"Layer {i + 1} weights (dim: {rounded_weights.shape}):\n {rounded_weights}")
            print(f"Layer {i + 1} biases (dim: {rounded_biases.shape}):\n {rounded_biases}")
        x = np.linspace(0, np.pi, 100)
        y = np.sin(x)
        rbf.train_nonlinear_optimization(x, y)
        rbf.plot(x, y)
        print("FINAL WEIGHTS")
        for i in range(len(rbf.weights)):
            rounded_weights = np.round(rbf.weights[i], 2)
            rounded_biases = np.round(rbf.biases[i], 2)
            print(f"Layer {i + 1} weights (dim: {rounded_weights.shape}):\n {rounded_weights}")
            print(f"Layer {i + 1} biases (dim: {rounded_biases.shape}):\n {rounded_biases}")


def q5():
    gammas = [0.1, 0.2, 0.4, 0.6, 0.8, 1]
    for gamma in gammas:
        rbf = RBF(epochs=2000, sse_threshold=0.001, lr=0.01, momentum=gamma)
        print("INITIAL WEIGHTS")
        for i in range(len(rbf.weights)):
            rounded_weights = np.round(rbf.weights[i], 2)
            rounded_biases = np.round(rbf.biases[i], 2)
            print(f"Layer {i + 1} weights (dim: {rounded_weights.shape}):\n {rounded_weights}")
            print(f"Layer {i + 1} biases (dim: {rounded_biases.shape}):\n {rounded_biases}")
        x = np.linspace(0, np.pi, 100)
        y = np.sin(x)
        rbf.train_nonlinear_optimization_with_momentum(x, y)
        rbf.plot(x, y)
        print("FINAL WEIGHTS")
        for i in range(len(rbf.weights)):
            rounded_weights = np.round(rbf.weights[i], 2)
            rounded_biases = np.round(rbf.biases[i], 2)
            print(f"Layer {i + 1} weights (dim: {rounded_weights.shape}):\n {rounded_weights}")
            print(f"Layer {i + 1} biases (dim: {rounded_biases.shape}):\n {rounded_biases}")


q4()
q5()