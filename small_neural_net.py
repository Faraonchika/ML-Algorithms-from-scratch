import numpy as np
import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def make_neural_net(train, n=0.25, gap=0.21):
    θ1 = np.array([0.8, 0.2])
    θ = np.array([[0.5, 0.2], [0.1, -0.3], [-0.1, 0.5]])
    error = 1000
    length = len(train)
    errors = 0
    while error > gap:
        for now in train:
            x = now[0]
            y = now[1]

            u = np.array([sigmoid(i) for i in np.dot(x, θ)])
            a = sigmoid(np.dot(u, θ1))

            errors += abs(y - a)

            delta = (a - y) * a * (1 - a)
            θ1 = θ1 - n * delta * u
            for i in range(len(θ)):
                for j in range(len(θ[0])):
                    θ[i, j] = θ[i, j] - n * (a - y) * a * (1 - a) * θ1[j] * u[j] * (1 - u[j]) * x[i]

        error = errors/length
        errors = 0
    
    return θ, θ1

def net_predict(weights, now, decision=0.5):
    θ = weights[0]
    θ1 = weights[1]
    u = np.array([sigmoid(i) for i in np.dot(now, θ)])
    a = sigmoid(np.dot(u, θ1))

    return 1 if a > decision else 0
