import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

def cub_function(x, a, b, c, d):
    return a * x ** 3 + b * x ** 2 + c * x + d

def sinusoidal_function(x):
    return x * np.sin(2 * np.pi * x)

def generate_sample(x, eps, error_type, function_type, parameters=None):
    N = x.shape[0]
    if error_type == "uniform":
        error = np.random.uniform(-eps, eps, N)
    elif error_type == "normal":
        error = np.random.normal(0, eps / 3, N)

    if function_type == "cub":
        if parameters is None:
            parameters = np.random.uniform(-3, 3, 4)
        a, b, c, d = parameters
        y = cub_function(x.ravel(), a, b, c, d) + error
        func = lambda t: cub_function(t, a, b, c, d)
    else:
        y = sinusoidal_function(x.ravel()) + error
        func = sinusoidal_function
        parameters = None

    return y, func, parameters

def plot_poly_regression(x, y_uniform, y_normal, true_func, eps, function_type, degrees, params):
    x_test = np.linspace(-1, 1, 100).reshape(-1, 1)
    y_true = true_func(x_test.ravel())

    plt.figure(figsize=(12, 8))
    plt.plot(x_test, y_true, color="black", label="Функция")

    for degree in degrees:
        model_uniform = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model_uniform.fit(x, y_uniform)
        y_pred_uniform = model_uniform.predict(x_test)
        plt.plot(x_test, y_pred_uniform, color="blue", label=f"Полиномиальная регрессия {degree}-й степени с равномерным распределением ошибок")

        model_normal = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model_normal.fit(x, y_normal)
        y_pred_normal = model_normal.predict(x_test)
        plt.plot(x_test, y_pred_normal, color="green", label=f"Полиномиальная регрессия {degree}-й степени с нормальным распределение ошибок")

    plt.scatter(x, y_uniform, color="blue", alpha=0.5, label="Выборка с равномерной ошибкой")
    plt.scatter(x, y_normal, color="green", alpha=0.5, label="Выборка с нормальной ошибкой")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Полиномиальная регрессия (eps={eps}, function={function_type})")

    if params is not None:
        plt.suptitle(f"Параметры полинома: a={params[0]:.2f}, b={params[1]:.2f}, c={params[2]:.2f}, d={params[3]:.2f}",
                     fontsize=10)

    plt.legend()
    plt.grid()
    plt.show()

N = 25
eps_values = [0.1, 0.7]
degrees = [1, 3, 5]
function_types = ["cub", "sinusoidal"]

for eps in eps_values:
    for degree in degrees:
        for function_type in function_types:
            x = np.random.uniform(-1, 1, N).reshape(-1, 1)
            y_uniform, true_func, params = generate_sample(x, eps, "uniform", function_type)
            y_normal, _, _ = generate_sample(x, eps, "normal", function_type, params)
            plot_poly_regression(x, y_uniform, y_normal, true_func, eps, function_type, [degree], params)
