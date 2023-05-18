import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from Simulation import simulation

# Call function to simulate a walking gait:
sim = simulation(-0.2572, 0.9342, 0.8426, 0.5573)

# Create "empty" figure and add axes
fig = plt.figure()
ax = plt.axes(xlim=(0, 3), ylim=(0, 1.5))
ax.grid()
plt.xlabel('Horizontal Distance (m)')
plt.ylabel('Vertical Distance (m)')
plt.title('Animation for Walking without Damping')

# Create the two line objects (one for each leg) which change during the animation
line1, = ax.plot([], [], lw=2)
line2, = ax.plot([], [], lw=2)


# Function to create the base frame upon which the animation takes place
def init():
    line1.set_data([], [])
    line2.set_data([], [])
    return line1, line2,  # returns the line objects that are to be plotted while they change after each frame


# Animation function (takes a single parameter, "i", for each frame)
def animate(i):
    x1 = np.array([sim[6, i], sim[1, i]])
    y1 = np.array([sim[7, i], sim[2, i]])

    x2 = np.array([sim[8, i], sim[1, i]])
    y2 = np.array([sim[9, i], sim[2, i]])

    line1.set_data(x1, y1)
    line2.set_data(x2, y2)
    return line1, line2,


# Create animation object
ani = animation.FuncAnimation(fig, animate, frames=sim.shape[1], interval=30, init_func=init)

# Save the animation as a ".gif"
"""
writer_gif = animation.PillowWriter(fps=30)
ani.save('Walking_Damping.gif', writer=writer_gif)
"""

plt.show()