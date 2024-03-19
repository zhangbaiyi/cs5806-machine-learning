
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

np.random.seed(5806)
pd.set_option('float_format', '{:.2f}'.format)

font1 = {'family':'serif', 'color':'darkred', 'size':30}
font2 = {'family':'serif', 'color':'blue', 'size':25}

class AnimalFactory:
    def __init__(self):
        self.prototype = np.array([[1, 4], [1, 5], [2, 4], [2, 5], [3, 1], [3, 2], [4, 1], [4, 2]]).reshape(8, 2)
        self.target = np.array([0,0,0,0, 1, 1, 1, 1])

class FourClassFactory:
    def __init__(self):
        self.prototype = np.array([[1, 4], [1, 5], [2, 4], [2, 5], [3, 1], [3, 2], [4, 1], [4, 2], [-1,1], [-1,2], [0,1], [0,2], [2,0], [2,-1], [3,0], [3,-1]]).reshape(16, 2)
        self.target = np.array([[0,0],[0,0],[0,0],[0,0], [0,1], [0,1], [0,1], [0,1], [1,0], [1,0], [1,0], [1,0], [1,1], [1,1], [1,1], [1,1]])


def plot_classes(prototype, target):
    df = pd.DataFrame(prototype, columns=['x', 'y'])
    df['target'] = target

    class_0_cord = df[df['target'] == 0].copy()
    class_0_cord.drop('target', axis=1, inplace=True)
    class_1_cord = df[df['target'] == 1].copy()
    class_1_cord.drop('target', axis=1, inplace=True)
    plt.figure(figsize=(8,8))
    plt.grid()
    plt.scatter(class_0_cord['x'], class_0_cord['y'], marker='s', c='r', s = 200, label='Rabbit (0) - I')
    plt.scatter(class_1_cord['x'], class_1_cord['y'], marker='d', c='b', s = 200, label='Bear (1) - II')
    plt.title('Prototypes', fontdict=font1)
    plt.xlabel('x', fontdict=font2)
    plt.ylabel('y', fontdict=font2)
    plt.axis('equal')
    plt.xlim(-2, 7)
    plt.ylim(-2, 7)
    plt.legend(markerscale=0.5)
    ret = plt
    return ret

def plot_classes_small_marker(prototype, target):
    df = pd.DataFrame(prototype, columns=['x', 'y'])
    df['target'] = target

    class_0_cord = df[df['target'] == 0].copy()
    class_0_cord.drop('target', axis=1, inplace=True)
    class_1_cord = df[df['target'] == 1].copy()
    class_1_cord.drop('target', axis=1, inplace=True)
    plt.figure(figsize=(8,8))
    plt.grid()
    plt.scatter(class_0_cord['x'], class_0_cord['y'], marker='s', c='r', s = 10, label='Rabbit (0)')
    plt.scatter(class_1_cord['x'], class_1_cord['y'], marker='d', c='b', s = 10, label='Bear (1)')
    plt.title('Prototypes', fontdict=font1)
    plt.xlabel('x', fontdict=font2)
    plt.ylabel('y', fontdict=font2)
    plt.axis('equal')
    plt.xlim(-2, 7)
    plt.ylim(-2, 7)
    plt.legend()
    ret = plt
    return ret

def plot_decision_boundary(W, b):
    x = np.linspace(-2, 7, 100)
    y = -(W[0, 0] * x + b[0, 0]) / W[0, 1]
    mid_x = (x.min() + x.max()) / 2
    mid_y = -(W[0, 0] * mid_x + b[0, 0]) / W[0, 1]
    norm = np.sqrt(W[0, 0] ** 2 + W[0, 1] ** 2)
    w_normalized = W / norm
    plt.plot(x, y, color='black', linewidth=4, label='Decision boundary')
    plt.quiver(mid_x, mid_y, w_normalized[0, 0], w_normalized[0, 1], angles='xy', scale_units='xy', scale=1, color='blue', width=0.005)
    plt.axis('equal')
    plt.xlim(-2, 7)
    plt.ylim(-2, 7)
    plt.title('Q2 Linearly Separable', fontdict=font1)
    plt.grid(True)
    plt.xticks(np.arange(-2, 8, 1))
    plt.yticks(np.arange(-2, 8, 1))
    plt.legend(markerscale=0.5)

def isConverged(E):
    return np.all(E == 0)

def hardlim(n):
    return 1 if n > 0 else 0

def perceptron_learning_algorithm(prototype, target):
    W = np.random.normal(0, 5, (1,2))
    b = np.random.normal(0, 5, (1,1))
    print('Initial weight and bias:\n')
    print('W =\n', np.array2string(W, formatter={'float_kind': '{0:.2f}'.format}))
    print('b =\n', np.array2string(b, formatter={'float_kind': '{0:.2f}'.format}))
    plot_classes(prototype, target)
    plot_decision_boundary(W, b)
    plt.savefig('initial_decision_boundary_sparse.png')
    plt.show()
    time.sleep(1.5)
    E = np.array([1]*8).reshape(8, 1)
    iterations = 0
    while not isConverged(E):
        iterations += 1
        for i in range(len(prototype)):
            x = prototype[i].reshape(1, 2)
            y = target[i]
            a = hardlim(np.dot(W, x.T) + b)
            e = y - a
            E[i] = e
            W = W + e * x
            b = b + e
        print('Weight and bias\n')
        print('W =\n', np.array2string(W, formatter={'float_kind': '{0:.2f}'.format}))
        print('b =\n', np.array2string(b, formatter={'float_kind': '{0:.2f}'.format}))
        plot_classes(prototype, target)
        plot_decision_boundary(W, b)
        plt.savefig('decision_boundary_' + str(iterations) + '.png')
        plt.show()
        time.sleep(1.5)

    print('Final weight and bias:\n')
    print('W =\n', np.array2string(W, formatter={'float_kind': '{0:.2f}'.format}))
    print('b =\n', np.array2string(b, formatter={'float_kind': '{0:.2f}'.format}))
    print('Testing the final weight and bias')
    for i in range(len(prototype)):
        x = prototype[i].reshape(1, 2)
        y = target[i]
        a = hardlim(np.dot(W, x.T) + b)
        print('prototype:', x, 'target (0: Class I, 1: Class II):', y, 'classification result:', a, 'error:', y-a)

def noisy_data_points(problem):
    df = pd.DataFrame(problem.prototype, columns=['x', 'y'])
    df['target'] = problem.target
    class_0_cord = df[df['target'] == 0].copy()
    class_0_cord.drop('target', axis=1, inplace=True)
    class_1_cord = df[df['target'] == 1].copy()
    class_1_cord.drop('target', axis=1, inplace=True)
    class_0_centroid = class_0_cord.mean()
    class_1_centroid = class_1_cord.mean()
    plot_classes(problem.prototype, problem.target)
    plt.title('Centroids', fontdict=font1)
    plt.scatter(class_0_centroid['x'], class_0_centroid['y'], marker='+', c='red', s = 200, label='Class 0 Centroid')
    plt.scatter(class_1_centroid['x'], class_1_centroid['y'], marker='+', c='blue', s = 200, label='Class 1 Centroid')
    plt.legend(markerscale=0.5)
    plt.savefig('centroids.png')
    plt.show()
    time.sleep(1.5)

    n = 800
    r_min, r_max = 0, 1.5
    # Class 0 Noisy Data Points Generating
    class_0_x_centroid = class_0_centroid['x']
    class_0_y_centroid = class_0_centroid['y']
    angles = np.random.uniform(0, 2*np.pi, n)
    radii = np.random.uniform(r_min, r_max, n)
    c0_x_points = class_0_x_centroid + radii * np.cos(angles)
    c0_y_points = class_0_y_centroid + radii * np.sin(angles)
    problem.prototype = np.append(problem.prototype, np.array([c0_x_points, c0_y_points]).T, axis=0)
    problem.target = np.append(problem.target, np.zeros(n))

    # Class 1 Noisy Data Points Generating
    class_1_x_centroid = class_1_centroid['x']
    class_1_y_centroid = class_1_centroid['y']
    angles = np.random.uniform(0, 2*np.pi, n)
    radii = np.random.uniform(r_min, r_max, n)
    c1_x_points = class_1_x_centroid + radii * np.cos(angles)
    c1_y_points = class_1_y_centroid + radii * np.sin(angles)
    problem.prototype = np.append(problem.prototype, np.array([c1_x_points, c1_y_points]).T, axis=0)
    problem.target = np.append(problem.target, np.ones(n))

    plot_classes_small_marker(problem.prototype, problem.target)
    plt.title('Noisy Data Points', fontdict=font1)
    plt.legend()
    plt.savefig('noisy_data_points.png')
    plt.show()
    time.sleep(1.5)

def optimum_decision_boundary(problem):
    print("Starting from random weight and bias")
    W = np.random.normal(0, 5, (1, 2))
    b = np.random.normal(0, 5, (1, 1))
    print('Initial weight and bias:\n')
    print('W =\n', np.array2string(W, formatter={'float_kind': '{0:.2f}'.format}))
    print('b =\n', np.array2string(b, formatter={'float_kind': '{0:.2f}'.format}))
    plot_classes_small_marker(problem.prototype, problem.target)
    plot_decision_boundary(W, b)
    plt.title('Initial Decision Boundary', fontdict=font1)
    plt.savefig('initial_decision_boundary.png')
    plt.show()
    time.sleep(1.5)
    E = np.array([1]*1608).reshape(1608, 1)
    iterations = 0
    while not isConverged(E):
        iterations += 1
        for i in range(len(problem.prototype)):
            x = problem.prototype[i].reshape(1, 2)
            y = problem.target[i]
            a = hardlim(np.dot(W, x.T) + b)
            e = y - a
            E[i] = e
            W = W + e * x
            b = b + e
        plot_classes_small_marker(problem.prototype, problem.target)
        plot_decision_boundary(W, b)
        plt.title('Q4 Linearly Separable', fontdict=font1)
        plt.savefig('optimum_decision_boundary_' + str(iterations) + '.png')
        plt.show()
        time.sleep(1.5)

    print('Final weight and bias:')
    print('W =\n', np.array2string(W, formatter={'float_kind': '{0:.2f}'.format}))
    print('b =\n', np.array2string(b, formatter={'float_kind': '{0:.2f}'.format}))
    print('Graphic Analysis of the final decision boundary')
    df = pd.DataFrame(problem.prototype, columns=['x', 'y'])
    df['target'] = problem.target
    class_0_cord = df[df['target'] == 0].copy()
    class_0_cord.drop('target', axis=1, inplace=True)
    class_1_cord = df[df['target'] == 1].copy()
    class_1_cord.drop('target', axis=1, inplace=True)
    class_0_centroid = class_0_cord.mean()
    class_1_centroid = class_1_cord.mean()

    distance_to_class_0 = np.dot(W, class_0_centroid) + b
    distance_to_class_1 = np.dot(W, class_1_centroid) + b

    # Print distances with 2-digit precision
    print(f'Distance to Class 0 centroid: {distance_to_class_0.item():.2f}')
    print(f'Distance to Class 1 centroid: {distance_to_class_1.item():.2f}')
def linear_inseperable_data_points(problem):
    df = pd.DataFrame(problem.prototype, columns=['x', 'y'])
    df['target'] = problem.target
    class_0_cord = df[df['target'] == 0].copy()
    class_0_cord.drop('target', axis=1, inplace=True)
    class_1_cord = df[df['target'] == 1].copy()
    class_1_cord.drop('target', axis=1, inplace=True)
    class_0_centroid = class_0_cord.mean()
    class_1_centroid = class_1_cord.mean()

    n = 800
    r_min, r_max = 0, 3
    # Class 0 Noisy Data Points Generating
    class_0_x_centroid = class_0_centroid['x']
    class_0_y_centroid = class_0_centroid['y']
    angles = np.random.uniform(0, 2 * np.pi, n)
    radii = np.random.uniform(r_min, r_max, n)
    c0_x_points = class_0_x_centroid + radii * np.cos(angles)
    c0_y_points = class_0_y_centroid + radii * np.sin(angles)
    problem.prototype = np.append(problem.prototype, np.array([c0_x_points, c0_y_points]).T, axis=0)
    problem.target = np.append(problem.target, np.zeros(n))

    # Class 1 Noisy Data Points Generating
    class_1_x_centroid = class_1_centroid['x']
    class_1_y_centroid = class_1_centroid['y']
    angles = np.random.uniform(0, 2 * np.pi, n)
    radii = np.random.uniform(r_min, r_max, n)
    c1_x_points = class_1_x_centroid + radii * np.cos(angles)
    c1_y_points = class_1_y_centroid + radii * np.sin(angles)
    problem.prototype = np.append(problem.prototype, np.array([c1_x_points, c1_y_points]).T, axis=0)
    problem.target = np.append(problem.target, np.ones(n))

    plot_classes_small_marker(problem.prototype, problem.target)
    plt.title('Non-linearly Separable Data Points', fontdict=font1)
    plt.grid(True)
    plt.xticks(np.arange(-2, 8, 1))
    plt.yticks(np.arange(-2, 8, 1))
    plt.legend()
    plt.savefig('linear_inseparable_data_points.png')
    plt.show()
    time.sleep(1.5)

def failed_perceptron_learning_rule(impossible):
    print("Starting from random weight and bias")
    W = np.random.normal(0, 5, (1, 2))
    b = np.random.normal(0, 5, (1, 1))
    print('Initial weight and bias:\n')
    print('W =\n', np.array2string(W, formatter={'float_kind': '{0:.2f}'.format}))
    print('b =\n', np.array2string(b, formatter={'float_kind': '{0:.2f}'.format}))
    plot_classes_small_marker(impossible.prototype, impossible.target)
    plot_decision_boundary(W, b)
    plt.title('Initial Decision Boundary', fontdict=font1)
    plt.savefig('initial_decision_boundary_nls.png')
    plt.show()
    time.sleep(1.5)
    E = np.array([1] * 1608).reshape(1608, 1)
    forever_loop = False
    iterations = 0
    while not isConverged(E) and not forever_loop:
        iterations += 1
        if iterations > 10:
            forever_loop = True
        for i in range(len(impossible.prototype)):
            x = impossible.prototype[i].reshape(1, 2)
            y = impossible.target[i]
            a = hardlim(np.dot(W, x.T) + b)
            e = y - a
            E[i] = e
            W = W + e * x
            b = b + e
        plot_classes_small_marker(impossible.prototype, impossible.target)
        plot_decision_boundary(W, b)
        plt.title('Q5: Non-linearly Separable', fontdict=font1)
        plt.savefig('linear_inseparable_' + str(iterations) + '.png')
        plt.show()
        time.sleep(1.5)
    if forever_loop:
        print('Failed to converge after 10 iterations')

def plot_four_classes(prototype, target):
    df = pd.DataFrame(prototype, columns=['x', 'y'])
    df['target_0'] = target[:, 0]
    df['target_1'] = target[:, 1]

    class_0_0_cord = df[(df['target_0'] == 0) & (df['target_1'] == 0)].copy()
    class_0_1_cord = df[(df['target_0'] == 0) & (df['target_1'] == 1)].copy()
    class_1_0_cord = df[(df['target_0'] == 1) & (df['target_1'] == 0)].copy()
    class_1_1_cord = df[(df['target_0'] == 1) & (df['target_1'] == 1)].copy()


    plt.figure(figsize=(8, 8))
    plt.grid()
    plt.scatter(class_0_0_cord['x'], class_0_0_cord['y'], marker='s', c='red', s=200, label='Class 0, 0 (I)')
    plt.scatter(class_0_1_cord['x'], class_0_1_cord['y'], marker='d', c='blue', s=200, label='Class 0, 1 (II)')
    plt.scatter(class_1_0_cord['x'], class_1_0_cord['y'], marker='*', c='green', s=200, label='Class 1, 0 (III)')
    plt.scatter(class_1_1_cord['x'], class_1_1_cord['y'], marker='o', c='cyan', s=200, label='Class 1, 1 (IV)')
    plt.title('Prototypes - 4 Classes', fontdict=font1)
    plt.xlabel('x', fontdict=font2)
    plt.ylabel('y', fontdict=font2)
    plt.axis('equal')
    plt.xlim(-2, 7)
    plt.ylim(-2, 7)
    plt.legend(markerscale=0.5)
    ret = plt
    return ret

def hardlim_array(n):
    return np.array([1 if i > 0 else 0 for i in n])

def perceptron_learning_algorithm_4_classes(prototype, target):
    W = np.random.normal(0, 5, (2, 2))
    b = np.random.normal(0, 5, (2, 1))
    print('Initial weight and bias:\n')
    print('W =\n', np.array2string(W, formatter={'float_kind': '{0:.2f}'.format}))
    print('b =\n', np.array2string(b, formatter={'float_kind': '{0:.2f}'.format}))
    plot_four_classes(prototype, target)
    plot_decision_boundary_4_classes(W, b)
    plt.savefig('initial_decision_boundary_4_classes.png')
    plt.show()
    time.sleep(1.5)
    E = np.array([[1],[1]]*16).reshape(16, 2)
    iterations = 0
    while not isConverged(E):
        iterations += 1
        for i in range(len(prototype)):
            x = prototype[i].reshape(1, 2)
            y = target[i]
            a = hardlim_array(np.dot(W, x.T) + b)
            e = np.array(y - a).reshape(2, 1)
            E[i] = e.T
            W = W + e @ x
            b = b + e
        print('W =\n', np.array2string(W, formatter={'float_kind': '{0:.2f}'.format}))
        print('b =\n', np.array2string(b, formatter={'float_kind': '{0:.2f}'.format}))
        plot_four_classes(prototype, target)
        plot_decision_boundary_4_classes(W, b)
        plt.title('Q6: Linearly Separable', fontdict=font1)
        plt.savefig('decision_boundary_4_classes_' + str(iterations) + '.png')
        plt.show()
        time.sleep(1.5)

    print('Final')
    print('W =\n', np.array2string(W, formatter={'float_kind': '{0:.2f}'.format}))
    print('b =\n', np.array2string(b, formatter={'float_kind': '{0:.2f}'.format}))
    print('Testing the final weight and bias')
    for i in range(len(prototype)):
        x = prototype[i].reshape(1, 2)
        y = target[i]
        a = hardlim_array(np.dot(W, x.T) + b)
        print('prototype:', x, 'target:', y, 'classification result:', a, 'error:', y-a)


def plot_decision_boundary_4_classes(W, b):
    x = np.linspace(-2, 7, 100)
    y1 = -(W[0, 0] * x + b[0, 0]) / W[0, 1]
    y2 = -(W[1, 0] * x + b[1, 0]) / W[1, 1]
    mid_x1 = (x.min() + x.max()) / 2
    mid_y1 = -(W[0, 0] * mid_x1 + b[0, 0]) / W[0, 1]
    mid_y2 = -(W[1, 0] * mid_x1 + b[1, 0]) / W[1, 1]
    norm1 = np.sqrt(W[0, 0] ** 2 + W[0, 1] ** 2)
    norm2 = np.sqrt(W[1, 0] ** 2 + W[1, 1] ** 2)
    w_normalized1 = W[0] / norm1
    w_normalized2 = W[1] / norm2
    plt.plot(x, y1, color='magenta', linewidth=4, label='Decision boundary 1')
    plt.plot(x, y2, color='magenta', linestyle='--', linewidth=4, label='Decision boundary 2')
    plt.quiver(mid_x1, mid_y1, w_normalized1[0], w_normalized1[1], angles='xy', scale_units='xy', scale=1, color='green', width=0.005)
    plt.quiver(mid_x1, mid_y2, w_normalized2[0], w_normalized2[1], angles='xy', scale_units='xy', scale=1, color='blue', width=0.005)
    plt.axis('equal')
    plt.xlim(-2, 7)
    plt.ylim(-2, 7)
    plt.title('Q6: Linearly Separable', fontdict=font1)
    plt.grid(True)
    plt.xticks(np.arange(-2, 8, 1))
    plt.yticks(np.arange(-2, 8, 1))
    plt.legend(markerscale=0.5)


def plot_classes_small_marker_4_classes(prototype, target):
    df = pd.DataFrame(prototype, columns=['x', 'y'])
    df['target_0'] = target[:, 0]
    df['target_1'] = target[:, 1]

    class_0_0_cord = df[(df['target_0'] == 0) & (df['target_1'] == 0)].copy()
    class_0_1_cord = df[(df['target_0'] == 0) & (df['target_1'] == 1)].copy()
    class_1_0_cord = df[(df['target_0'] == 1) & (df['target_1'] == 0)].copy()
    class_1_1_cord = df[(df['target_0'] == 1) & (df['target_1'] == 1)].copy()

    plt.figure(figsize=(8, 8))
    plt.grid()
    plt.scatter(class_0_0_cord['x'], class_0_0_cord['y'], marker='s', c='red', s=10, label='Class 0, 0')
    plt.scatter(class_0_1_cord['x'], class_0_1_cord['y'], marker='d', c='blue', s=10, label='Class 0, 1')
    plt.scatter(class_1_0_cord['x'], class_1_0_cord['y'], marker='*', c='green', s=10, label='Class 1, 0')
    plt.scatter(class_1_1_cord['x'], class_1_1_cord['y'], marker='o', c='cyan', s=10, label='Class 1, 1')
    plt.xlabel('x', fontdict=font2)
    plt.ylabel('y', fontdict=font2)
    plt.axis('equal')
    plt.xlim(-2, 7)
    plt.ylim(-2, 7)
    plt.legend()
    ret = plt
    return ret


def noisy_data_points_4_classes(problem):
    target = problem.target
    df = pd.DataFrame(problem.prototype, columns=['x', 'y'])
    df['target_0'] = target[:, 0]
    df['target_1'] = target[:, 1]

    class_0_0_cord = df[(df['target_0'] == 0) & (df['target_1'] == 0)].copy()
    class_0_1_cord = df[(df['target_0'] == 0) & (df['target_1'] == 1)].copy()
    class_1_0_cord = df[(df['target_0'] == 1) & (df['target_1'] == 0)].copy()
    class_1_1_cord = df[(df['target_0'] == 1) & (df['target_1'] == 1)].copy()

    class_0_0_centroid = class_0_0_cord.mean()
    class_0_1_centroid = class_0_1_cord.mean()
    class_1_0_centroid = class_1_0_cord.mean()
    class_1_1_centroid = class_1_1_cord.mean()

    n = 800
    r_min, r_max = 0, 1

    # Class 00 Noisy Data Points Generating
    class_0_0_x_centroid = class_0_0_centroid['x']
    class_0_0_y_centroid = class_0_0_centroid['y']
    angles = np.random.uniform(0, 2*np.pi, n)
    radii = np.random.uniform(r_min, r_max, n)
    c00_x_points = class_0_0_x_centroid + radii * np.cos(angles)
    c00_y_points = class_0_0_y_centroid + radii * np.sin(angles)
    problem.prototype = np.append(problem.prototype, np.array([c00_x_points, c00_y_points]).T, axis=0)
    target_to_append = np.array([[0, 0]]*n).reshape(n, 2)
    problem.target = np.vstack((problem.target, target_to_append))

    # Class 01 Noisy Data Points Generating
    class_0_1_x_centroid = class_0_1_centroid['x']
    class_0_1_y_centroid = class_0_1_centroid['y']
    angles = np.random.uniform(0, 2*np.pi, n)
    radii = np.random.uniform(r_min, r_max, n)
    c01_x_points = class_0_1_x_centroid + radii * np.cos(angles)
    c01_y_points = class_0_1_y_centroid + radii * np.sin(angles)
    problem.prototype = np.append(problem.prototype, np.array([c01_x_points, c01_y_points]).T, axis=0)
    target_to_append = np.array([[0, 1]]*n).reshape(n, 2)
    problem.target = np.vstack((problem.target, target_to_append))

    # Class 10 Noisy Data Points Generating
    class_1_0_x_centroid = class_1_0_centroid['x']
    class_1_0_y_centroid = class_1_0_centroid['y']
    angles = np.random.uniform(0, 2*np.pi, n)
    radii = np.random.uniform(r_min, r_max, n)
    c10_x_points = class_1_0_x_centroid + radii * np.cos(angles)
    c10_y_points = class_1_0_y_centroid + radii * np.sin(angles)
    problem.prototype = np.append(problem.prototype, np.array([c10_x_points, c10_y_points]).T, axis=0)
    target_to_append = np.array([[1, 0]]*n).reshape(n, 2)
    problem.target = np.vstack((problem.target, target_to_append))

    # Class 11 Noisy Data Points Generating
    class_1_1_x_centroid = class_1_1_centroid['x']
    class_1_1_y_centroid = class_1_1_centroid['y']
    angles = np.random.uniform(0, 2*np.pi, n)
    radii = np.random.uniform(r_min, r_max, n)
    c11_x_points = class_1_1_x_centroid + radii * np.cos(angles)
    c11_y_points = class_1_1_y_centroid + radii * np.sin(angles)
    problem.prototype = np.append(problem.prototype, np.array([c11_x_points, c11_y_points]).T, axis=0)
    target_to_append = np.array([[1, 1]]*n).reshape(n, 2)
    problem.target = np.vstack((problem.target, target_to_append))

    plot_classes_small_marker_4_classes(problem.prototype, problem.target)
    plt.title('Noisy Data Points - 4 Classes', fontdict=font1)
    plt.legend()
    plt.savefig('noisy_data_points_4_classes.png')
    plt.show()
    time.sleep(1.5)

    centroids = np.array([class_0_0_centroid, class_0_1_centroid, class_1_0_centroid, class_1_1_centroid])
    return centroids


def optimum_decision_boundary_4_classes(problem):
    print("Starting from random weight and bias\n")
    W = np.random.normal(0, 5, (2, 2))
    b = np.random.normal(0, 5, (2, 1))
    print('W =\n', np.array2string(W, formatter={'float_kind': '{0:.2f}'.format}))
    print('b =\n', np.array2string(b, formatter={'float_kind': '{0:.2f}'.format}))
    plot_classes_small_marker_4_classes(problem.prototype, problem.target)
    plot_decision_boundary_4_classes(W, b)
    plt.show()
    time.sleep(1.5)
    n = problem.prototype.shape[0]
    E = np.array([[1], [1]] * n).reshape(n, 2)
    iterations = 0
    while not isConverged(E):
        iterations += 1
        for i in range(n):
            x = problem.prototype[i].reshape(1, 2)
            y = problem.target[i]
            a = hardlim_array(np.dot(W, x.T) + b)
            e = np.array(y - a).reshape(2, 1)
            E[i] = e.T
            W = W + e @ x
            b = b + e
        print('W =\n', np.array2string(W, formatter={'float_kind': '{0:.2f}'.format}))
        print('b =\n', np.array2string(b, formatter={'float_kind': '{0:.2f}'.format}))
        plot_classes_small_marker_4_classes(problem.prototype, problem.target)
        plot_decision_boundary_4_classes(W, b)
        plt.title('Q7: Linearly Separable', fontdict=font1)
        plt.savefig('optimum_decision_boundary_4_classes_' + str(iterations) + '.png')
        plt.show()
        time.sleep(1.5)

    print('Final: \n')
    print('W =\n', np.array2string(W, formatter={'float_kind': '{0:.2f}'.format}))
    print('b =\n', np.array2string(b, formatter={'float_kind': '{0:.2f}'.format}))

    return W, b


def testCentroids(W, b, centroids):
    class_00_centroids = centroids[0][:2]
    class_01_centroids = centroids[1][:2]
    class_10_centroids = centroids[2][:2]
    class_11_centroids = centroids[3][:2]
    w0 = W[0]
    w1 = W[1]
    b0 = b[0]
    b1 = b[1]
    print('Testing the final weight and bias with centroids')
    dis_class00_w0 = np.dot(w0, class_00_centroids) + b0
    dis_class00_w1 = np.dot(w1, class_00_centroids) + b1
    dis_class01_w0 = np.dot(w0, class_01_centroids) + b0
    dis_class01_w1 = np.dot(w1, class_01_centroids) + b1
    dis_class10_w0 = np.dot(w0, class_10_centroids) + b0
    dis_class10_w1 = np.dot(w1, class_10_centroids) + b1
    dis_class11_w0 = np.dot(w0, class_11_centroids) + b0
    dis_class11_w1 = np.dot(w1, class_11_centroids) + b1
    print(f'Distance to Class 00 centroid with W0: {dis_class00_w0.item():.2f}')
    print(f'Distance to Class 01 centroid with W0: {dis_class01_w0.item():.2f}')
    print(f'Distance to Class 10 centroid with W0: {dis_class10_w0.item():.2f}')
    print(f'Distance to Class 11 centroid with W0: {dis_class11_w0.item():.2f}')
    sum_w0 = dis_class00_w0.item() + dis_class01_w0.item() + dis_class10_w0.item() + dis_class11_w0.item()
    print(f'Sum of distances = {sum_w0:.2f}')

    print(f'Distance to class 00 centroid with W1: {dis_class00_w1.item():.2f}')
    print(f'Distance to class 01 centroid with W1: {dis_class01_w1.item():.2f}')
    print(f'Distance to class 10 centroid with W1: {dis_class10_w1.item():.2f}')
    print(f'Distance to class 11 centroid with W1: {dis_class11_w1.item():.2f}')
    sum_w1 = dis_class00_w1.item() + dis_class01_w1.item() + dis_class10_w1.item() + dis_class11_w1.item()
    print(f'Sum of distances = {sum_w1:.2f}')

    print("Orthogonality test")
    dot_product = np.dot(w0, w1)
    print(f'Dot product of W0 and W1: {dot_product.item():.2f}')
    print('The dot product is close to zero, so W0 and W1 are close to orthogonal')


def main():
    problem = AnimalFactory()
    impossible = AnimalFactory()
    plot_classes(problem.prototype, problem.target)
    plt.savefig('prototypes.png')
    plt.show()
    time.sleep(1.5)
    perceptron_learning_algorithm(problem.prototype, problem.target)
    noisy_data_points(problem)
    optimum_decision_boundary(problem)
    linear_inseperable_data_points(impossible)
    failed_perceptron_learning_rule(impossible)

    problem = FourClassFactory()
    plot_four_classes(problem.prototype, problem.target)
    plt.savefig('prototypes_4_classes.png')
    plt.show()
    time.sleep(1.5)
    perceptron_learning_algorithm_4_classes(problem.prototype, problem.target)
    centroids = noisy_data_points_4_classes(problem)
    W, b = optimum_decision_boundary_4_classes(problem)
    testCentroids(W, b, centroids)









if __name__ == '__main__':
    main()
