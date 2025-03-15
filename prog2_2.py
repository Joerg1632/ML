import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from sklearn.metrics import confusion_matrix

def generate_linear(n):
    x = np.random.uniform(-1, 1, (n, 2))
    y = (x[:, 0] + x[:, 1] > 0).astype(int)
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def step_function(x):
    return torch.where(x >= 0, torch.tensor(1.0), torch.tensor(0.0))

class PerceptronStep(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(input_dim))
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return step_function(torch.sum(x * self.weights, dim=1) + self.bias)

    def train_perceptron(self, x, y, epochs=100):
        for epoch in range(epochs):
            for i in range(len(x)):
                y_pred = self.forward(x[i].unsqueeze(0))
                error = y[i] - y_pred
                self.weights.data += error * x[i]
                self.bias.data += error

class PerceptronSigmoid(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.activation(self.linear(x)).squeeze()

def train_perceptron_sigmoid(model, x, y, epochs=100, lr=0.1):
    optimizer = optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    for _ in range(epochs):
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()


n_samples = 200
x_train, y_train = generate_linear(n_samples)
x_test, y_test = generate_linear(n_samples)

perceptron_step = PerceptronStep(2)
start_time = time.time()
perceptron_step.train_perceptron(x_train, y_train, epochs=10)
step_time = time.time() - start_time

y_pred_step = perceptron_step.forward(x_test)

perceptron_sigmoid = PerceptronSigmoid(2)
start_time = time.time()
train_perceptron_sigmoid(perceptron_sigmoid, x_train, y_train, epochs=100, lr=0.1)
sigmoid_time = time.time() - start_time

y_pred_sigmoid = perceptron_sigmoid(x_test).round()

print("Confusion Matrix (Step Function):\n", confusion_matrix(y_test, y_pred_step.detach().numpy()))
print("Confusion Matrix (Sigmoid):\n", confusion_matrix(y_test, y_pred_sigmoid.detach().numpy()))

print(f"Training time (Step Function): {step_time:.4f} sec")
print(f"Training time (Sigmoid): {sigmoid_time:.4f} sec")