import numpy as np
import matplotlib.pyplot as plt

x = [1,2,3,4,5]
y = [2,3,6,9,11]

plt.scatter(x, y)

y = np.array(y)
x = np.array(x)

# gradient(m) - sum((xi - mean of x) * (yi - mean of y) / sum(xi - mean of x)²)
# y-intercept(c) - mean of y - (m) * mean of x

mx = x.mean()
my = y.mean()
m = np.sum(((x - mx) * (y - my)) / np.sum((x - mx)**2))
print(m)

c = my - m * mx
print(c)

y1 = m*x + c
print(y)


plt.plot(x,y1)
plt.show()