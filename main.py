import numpy as np
import matplotlib.pyplot as plt


def generate_samples(N, epsilon_0, error_type, function_type):
    x = np.clip(np.random.normal(0, 0.5, N), -1, 1)
    # Generation coef for polynomial func
    if function_type == 'polynomial':
        a, b, c, d = np.random.uniform(-3, 3, 4)
        f_x = a * x ** 3 + b * x ** 2 + c * x + d
    elif function_type == 'trig':
        f_x = x * np.sin(2 * np.pi * x)
    else:
        raise ValueError("Unknown function type")

    # Generation error
    if error_type == 'uniform':
        epsilon = np.random.uniform(-epsilon_0, epsilon_0, N)
    elif error_type == 'normal':
        epsilon = np.random.normal(0, epsilon_0 / np.sqrt(3), N)  # Масштабируем стандартное отклонение
    else:
        raise ValueError("Unknown error type")

    # Generate y
    y = f_x + epsilon

    return x, y, f_x


# Parametrs
N = 100  # Num dots
epsilon_0 = 0.2  # Error

# Generation and visual
for error_type in ['uniform', 'normal']:
    for function_type in ['polynomial', 'trig']:
        x, y, f_x = generate_samples(N, epsilon_0, error_type, function_type)

        plt.figure(figsize=(6, 4))
        plt.scatter(x, y, label='Noisy data', alpha=0.6)
        plt.scatter(x, f_x, label='True function', color='red', s=10)
        plt.legend()
        plt.title(f"Function: {function_type}, Error: {error_type}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()
