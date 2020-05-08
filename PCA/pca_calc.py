x = [2.5, 0.5, 2.2, 1.9, 3.1, 2.3, 2, 1, 1.5, 1.1]
y = [2.4, 0.7, 2.9, 2.2, 3.0, 2.7, 1.6, 1.1, 1.6, 0.9]
import numpy as np

mean_x = np.mean(x)
mean_y = np.mean(y)

result_x = [round(x[i] - mean_x, 2) for i in range(len(x))]
result_y = [round(y[i] - mean_y, 2) for i in range(len(y))]
# print(result_x)
# print(result_y)
z = np.vstack((result_x, result_y))
print(z)
cov = np.cov(z)
print(cov)

