# import numpy as np
# import matplotlib.pyplot as plt
#
# def cub_function(x, a, b, c, d):
#     return a * x ** 3 + b * x ** 2 + c * x + d
#
# def sinusoidal_function(x):
#     return x * np.sin(2 * np.pi * x)
#
# def generate_sample(N, eps, error_type, function_type):
#     x = np.random.uniform(-1, 1, N)
#
#     if error_type == "uniform":
#         error = np.random.uniform(-eps, eps, N)
#     elif error_type == "normal":
#         error = np.clip(np.random.normal(0, eps / 2, N), -eps, eps)
#
#     if function_type == "cub":
#         a, b, c, d = np.random.uniform(-3, 3, 4)
#         y = cub_function(x, a, b, c, d) + error
#         func = lambda t: cub_function(t, a, b, c, d)
#     else:
#         y = sinusoidal_function(x) + error
#         func = sinusoidal_function
#
#     return x, y, func
#
#
# def plot_sample(x, y, true_func, eps, error_type, function_type):
#     x_plot = np.linspace(-1, 1, 100)
#     y_true = true_func(x_plot)
#
#     plt.figure(figsize=(7, 5))
#     plt.scatter(x, y, color='blue', label='Выборка', alpha=0.7)
#     plt.plot(x_plot, y_true, color='red', label='Функция')
#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.title(f"Функция: {function_type}, Ошибка: {error_type}, eps={eps}")
#     plt.legend()
#     plt.grid(True)
#     plt.show()
#
#
# N = 50
# eps_values = [0.1, 0.5]
# error_types = ["uniform", "normal"]
# function_types = ["cub", "sinusoidal"]
#
# for function_type in function_types:
#     for error_type in error_types:
#         for eps in eps_values:
#             x, y, true_func = generate_sample(N, eps, error_type, function_type)
#             plot_sample(x, y, true_func, eps, error_type, function_type)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline


def cub_function(x, a, b, c, d):
    return a * x ** 3 + b * x ** 2 + c * x + d

def sinusoidal_function(x):
    return x * np.sin(2 * np.pi * x)

def generate_sample(N, eps, error_type, function_type):
    x = np.random.uniform(-1, 1, N).reshape(-1, 1)

    if error_type == "uniform":
        error = np.random.uniform(-eps, eps, N)
    elif error_type == "normal":
        error = np.clip(np.random.normal(0, eps / 3, N), -eps, eps)

    if function_type == "cub":
        a, b, c, d = np.random.uniform(-3, 3, 4)
        y = cub_function(x.ravel(), a, b, c, d) + error
        func = lambda t: cub_function(t, a, b, c, d)
        params = (a, b, c, d)
    else:
        y = sinusoidal_function(x.ravel()) + error
        func = sinusoidal_function
        params = None

    return x, y, func, params


def plot_poly_regression(x, y_uniform, y_normal, true_func, eps, function_type, degrees, params):
    x_test = np.linspace(-1, 1, 100).reshape(-1, 1)
    y_true = true_func(x_test.ravel())

    plt.figure(figsize=(12, 8))
    plt.plot(x_test, y_true, color="black", label="Функция")

    for degree in degrees:
        model_uniform = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model_uniform.fit(x, y_uniform)
        y_pred_uniform = model_uniform.predict(x_test)
        plt.plot(x_test, y_pred_uniform,color = "blue", label=f"Полиномиальная регрессия {degree}-й степени с равномерной ошибкой")

        model_normal = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model_normal.fit(x, y_normal)
        y_pred_normal = model_normal.predict(x_test)
        plt.plot(x_test, y_pred_normal,color = "green", label=f"Полиномиальная регрессия {degree}-й степени с нормальной ошибкой")

    plt.scatter(x, y_uniform, color="blue", alpha=0.5, label="Выборка с равномерной ошибкой")
    plt.scatter(x, y_normal, color="green", alpha=0.5, label="Выборка с нормальной ошибкой")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Полиномиальная регрессия (eps={eps}, function={function_type})")
    if params:
        plt.suptitle(f"Параметры полинома: a={params[0]:.2f}, b={params[1]:.2f}, c={params[2]:.2f}, d={params[3]:.2f}",
                     fontsize=10)
    plt.legend()
    plt.grid()
    plt.show()


N = 10
eps_values = [0.1, 0.5]
degrees = [1, 3, 5, 7, 10]
function_types = ["cub", "sinusoidal"]

for eps in eps_values:
    for degree in degrees:
        for function_type in function_types:
            x, y_uniform, true_func, params = generate_sample(N, eps, "uniform", function_type)
            _, y_normal, _, _ = generate_sample(N, eps, "normal", function_type)
            plot_poly_regression(x, y_uniform, y_normal, true_func, eps, function_type, [degree], params)