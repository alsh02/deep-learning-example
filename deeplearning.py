import numpy as np
import matplotlib.pyplot as plt

def layer(x, w):
    """Estimate weighted sum"""
    print(f"input shape: {x.shape}")
    print(f"weight shape: {w.shape}")
    result = np.matmul(w, x)
    result = activation(result)
    return sum(result)

def activation(x):
    """logistic function(non-linear)"""
    return 1 / (1 + np.exp(-x))

def loss_function(actual, predicted):
    """Binary cross entropy loss"""
    n = len(actual) # always same as "len(predicted)"
    pred_log = np.log(predicted)
    return (actual.T.dot(pred_log) + (1 - actual).T.dot(pred_log)) / n * (-1)

x = np.array([[3, 2], [5, 10], [10, 10], [8, 2], [1, 10], [4, 7], [1, 2]])
y = np.array([0, 1, 1, 0, 0, 1, 0])
w = np.array([[0.6, 0.2, 0.2], [0.3, 0.2, 0.5]]).T
print(f"input matrix: \n{x}")
print(f"weight matrix: \n{w}")

for i, label in enumerate(y):
    if label == 0:
        plt.scatter(x[i][0], x[i][1], color='r', label='fail')
    else:
        plt.scatter(x[i][0], x[i][1], color='g', label='pass')

# Plot scatter of data to see data distribution
plt.grid(True)
plt.legend()
plt.show()

# Training process
# 1 hidden layer, 3 nodes
pred = []
for data in x:
    pred.append(layer(data, w))
print(f"prediction: \n{pred}")    
    
loss = loss_function(y, pred)
print(loss)
    
# Test data
new_x = np.array([4, 5])