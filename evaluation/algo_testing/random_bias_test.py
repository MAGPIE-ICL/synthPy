import numpy as np
import matplotlib.pyplot as plt

length = 10000

x = np.arange(0, length, 1)
y = np.random.rand(length) + np.random.rand(length)




r = y.copy()
r[r > 1] = 2 - r[r > 1]

fig1, ax1 = plt.subplots()

ax1.scatter(x, r, label="Data Points", s = 1)

plt.xlabel("Data point")
plt.ylabel("Randomly allocated radial distance from central axis")
plt.title("Testing bias in random plots - Sorting correction")

plt.legend()



fig2, ax2 = plt.subplots()

ax2.scatter(x, y / 2, label="Data Points", s = 1)

plt.xlabel("Data point")
plt.ylabel("Randomly allocated radial distance from central axis")
plt.title("Testing bias in random plots - Half correction")

plt.legend()



fig3, ax3 = plt.subplots()

ax3.scatter(x, np.random.rand(length), label="Data Points", s = 1)

plt.xlabel("Data point")
plt.ylabel("Randomly allocated radial distance from central axis")
plt.title("Testing bias in random plots - Different generation")

plt.legend()



fig4, ax4 = plt.subplots()

ax4.scatter(x, y, label="Data Points", s = 1)

plt.xlabel("Data point")
plt.ylabel("Randomly allocated radial distance from central axis")
plt.title("Testing bias in random plots - No correction")

plt.legend()



'''
t = 2*np.random.rand(length)# - 1.0

fig5, ax5 = plt.subplots()

ax5.scatter(x, t, label="Data Points", s = 1)

plt.xlabel("Data point")
plt.ylabel("Randomly allocated from center of square")
plt.title("Testing bias in random plots - No correction, just checking")

plt.legend()
'''

plt.show()