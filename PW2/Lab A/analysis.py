import numpy as np
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt

data = np.loadtxt("freefall.csv", delimiter=",", skiprows=1)

t = data[:, 0]
y = data[:, 1]

velocity = np.gradient(y, t)
acceleration = np.gradient(velocity, t)

print("Mean acceleration:", np.mean(acceleration))
print("Standard deviation:", acceleration.std())

recovered_velocity = cumulative_trapezoid(acceleration, t, initial=0) + velocity[0]
recovered_position = cumulative_trapezoid(recovered_velocity, t, initial=0) + y[0]

difference = np.abs(recovered_position - y)
print("Largest difference:", np.max(difference))

fig, ax = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

ax[0].plot(t, y)
ax[0].set_ylabel("Position (m)")

ax[1].plot(t, velocity)
ax[1].set_ylabel("Velocity (m/s)")

ax[2].plot(t, acceleration)
ax[2].axhline(-9.81, linestyle="--")
ax[2].set_ylabel("Acceleration (m/s²)")
ax[2].set_xlabel("Time (s)")

plt.tight_layout()
plt.savefig("motion.png")

# Bonus: 2D trajectory

data2 = np.loadtxt("trajectory.csv", delimiter=",", skiprows=1)

t2 = data2[:, 0]
x = data2[:, 1]
y2 = data2[:, 2]

vx = np.gradient(x, t2)
vy = np.gradient(y2, t2)

speed = np.sqrt(vx**2 + vy**2)

plt.figure()
plt.plot(x, y2)
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("2D Trajectory")
plt.savefig("trajectory.png")

plt.figure()
plt.plot(t2, speed)
plt.xlabel("Time (s)")
plt.ylabel("Speed (m/s)")
plt.title("Speed vs Time")
plt.savefig("speed.png")