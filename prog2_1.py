import numpy as np
import matplotlib.pyplot as plt

def generate_linear(n):
    x = np.random.uniform(-1, 1, (n, 2))
    y = (x[:, 0] + x[:, 1] > 0).astype(int)
    return x, y

def add_noise(x, noise_type="gaussian", sigma=0.1, epsilon=0.1):
    if noise_type == "gaussian":
        return x + np.random.normal(0, sigma, x.shape)
    elif noise_type == "uniform":
        return x + np.random.uniform(-epsilon, epsilon, x.shape)
    return x

def generate_xor(n):
    x = np.random.uniform(-1, 1, (n, 2))
    y = ((x[:, 0] * x[:, 1]) > 0).astype(int)
    return x, y

def generate_circles(n):
    r = np.sqrt(np.random.uniform(0, 1, n))
    theta = np.random.uniform(0, 2 * np.pi, n)
    x = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
    y = (r > 0.5).astype(int)
    return x, y

def generate_spiral(n, noise=0.1):
    t = np.linspace(0, 4 * np.pi, n // 2)
    x1 = np.stack([t * np.cos(t), t * np.sin(t)], axis=1) / (4 * np.pi)
    x2 = np.stack([-t * np.cos(t), -t * np.sin(t)], axis=1) / (4 * np.pi)
    x = np.vstack([x1, x2])
    y = np.hstack([np.zeros(n // 2), np.ones(n // 2)]).astype(int)
    return x, y

def plot_data(x, y, title):
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap='bwr', alpha=0.6)
    plt.title(title)
    plt.show()

n_samples = 200
x_linear, y_linear = generate_linear(n_samples)
plot_data(x_linear, y_linear, "Линейно разделимые данные")
x_linear_noisy = add_noise(x_linear)
plot_data(x_linear_noisy, y_linear, "Линейные данные с шумом")


x_xor, y_xor = generate_xor(n_samples)
plot_data(x_xor, y_xor, "XOR")
x_xor_noisy = add_noise(x_xor)
plot_data(x_xor_noisy, y_xor, "Xor данные с шумом")


x_circles, y_circles = generate_circles(n_samples)
plot_data(x_circles, y_circles, "Кольца")
x_circles_noisy = add_noise(x_linear)
plot_data(x_circles_noisy, y_circles, "Кольцевые данные с шумом")


x_spiral, y_spiral = generate_spiral(n_samples)
plot_data(x_spiral, y_spiral, "Спираль")
x_spiral_noisy = add_noise(x_spiral)
plot_data(x_spiral_noisy, y_spiral, "Спиральные данные с шумом")
