import numpy as np
import matplotlib.pyplot as plt

def cub_function(x, a, b, c, d):
    return a * x ** 3 + b * x ** 2 + c * x + d

def sinusoidal_function(x):
    return x * np.sin(2 * np.pi * x)

def generate_sample(N, eps, error_type, function_type):
    x = np.random.uniform(-1, 1, N)

    if error_type == "uniform":
        error = np.random.uniform(-eps, eps, N)
    elif error_type == "normal":
        error = np.clip(np.random.normal(0, eps / 2, N), -eps, eps)

    if function_type == "cub":
        a, b, c, d = np.random.uniform(-3, 3, 4)
        y = cub_function(x, a, b, c, d) + error
        func = lambda t: cub_function(t, a, b, c, d)
    else:
        y = sinusoidal_function(x) + error
        func = sinusoidal_function

    return x, y, func


def plot_sample(x, y, true_func, eps, error_type, function_type):
    x_plot = np.linspace(-1, 1, 100)
    y_true = true_func(x_plot)

    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, color='blue', label='Выборка', alpha=0.7)
    plt.plot(x_plot, y_true, color='red', label='Функция')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f"Функция: {function_type}, Ошибка: {error_type}, eps={eps}")
    plt.legend()
    plt.grid(True)
    plt.show()


N = 50
eps_values = [0.1, 0.5]
error_types = ["uniform", "normal"]
function_types = ["cub", "sinusoidal"]

for function_type in function_types:
    for error_type in error_types:
        for eps in eps_values:
            x, y, true_func = generate_sample(N, eps, error_type, function_type)
            plot_sample(x, y, true_func, eps, error_type, function_type)