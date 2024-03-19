
import matplotlib.pyplot as plt
import numpy as np

#%%
x = np.linspace(-10, 10, 500)
F1 = -1.5*(x+0.5)**2 - 0.25
F2 = 0.006 + 2.88 * (x-1.1) + 0.5*6.6*(x-1.1)**2
O = x**3 - 0.75*x-0.5

plt.plot(x, F1, label='2nd at x* = -0.5')
plt.plot(x, F2, label='2nd at x* = 1.1')
plt.plot(x, O, label='Original')
plt.axvline(x=-0.5, color='r', linestyle='--')
plt.axvline(x=1.1, color='r', linestyle='--')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Original and 2nd order approximations')
plt.legend()
plt.show()

#%%
from mpl_toolkits.mplot3d import Axes3D
x1 = np.linspace(-1, 1.25, 500)
x2 = np.linspace(-1, 1.25, 500)
X1, X2 = np.meshgrid(x1, x2)
F = np.exp(2*X1**2 + 2*X2**2 + X1 - 5*X2 + 10)

fig = plt.figure(figsize=(12, 6))

ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.plot_surface(X1, X2, F, cmap='viridis')
ax.set_title('3D Plot')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('F')

ax2 = fig.add_subplot(1, 2, 2)
contour = ax2.contourf(X1, X2, F, cmap='viridis')
fig.colorbar(contour)
ax2.set_title('Contour Plot')
ax2.set_xlabel('X1')
ax2.set_ylabel('X2')

plt.show()


#%%
x = np.linspace(-10, 10, 500)
y = np.linspace(-10, 10, 500)
F = (x + y)**4 - 12*x*y + x + y + 1
delta_Fx = (4*(x + y)**3 - 12 * y + 1)
delta_Fy = (4*(x + y)**3 - 12 * x + 1)
delta_F = np.vstack((delta_Fx, delta_Fy))
delta_2_F = np.array([[12*(x + y)**2, 12*(x + y)**2 - 12], [12*(x + y)**2 - 12, 12*(x + y)**2]])