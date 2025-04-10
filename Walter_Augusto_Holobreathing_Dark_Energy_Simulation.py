import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Grid parameters
N = 200
x = np.linspace(0, 10, N)
y = np.linspace(0, 10, N)
X, Y = np.meshgrid(x, y)

# Simulation parameters
epsilon = 0.001  # Weak nonlinearity to allow expansion
dt = 0.01
steps = 500

# Initialize breathing field (small random perturbations)
phi = 0.01 * (np.random.rand(N, N) - 0.5)
phi_dot = np.zeros((N, N))

# Breathing equation update function
def update(phi, phi_dot, dt, epsilon):
    phi_ddot = -np.sin(phi) * (1 + epsilon * phi**2)
    phi_dot += phi_ddot * dt
    phi += phi_dot * dt
    return phi, phi_dot

# Set up plotting
fig, ax = plt.subplots(figsize=(6, 6))
cax = ax.imshow(phi, cmap='viridis', origin='lower', extent=[0, 10, 0, 10])
fig.colorbar(cax)

# Animation update function
def animate(i):
    global phi, phi_dot
    for _ in range(5):  # Speed up by multiple updates per frame
        phi, phi_dot = update(phi, phi_dot, dt, epsilon)
    cax.set_array(phi)
    return [cax]

ani = FuncAnimation(fig, animate, frames=steps, interval=30, blit=True)

# Save animation as GIF
ani.save("Walter_Augusto_Holobreathing_Dark_Energy.gif", fps=30)

plt.show()
