import math

def sigmoid(x):
    output = 1 / 1 + math.exp(-x)
    return output

def activation(input, weight):
    holder = 0;
    for i , w in zip(input, weight):
        holder += i*w
    return sigmoid(holder)    


if __name__ == "__main__":
    inputs = [1.2, 4.6, -3.1]
    weights = [-1.6, 0.2, 1.34]
    output = activation(inputs, weights)
    print(output);