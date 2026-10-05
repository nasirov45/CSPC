import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("titration.csv", delimiter=",", skiprows=1)

V = data[:, 0]
pH = data[:, 1]

slope = np.gradient(pH, V)

index = np.argmax(slope)
equivalence = V[index]

print("Equivalence point:", equivalence, "mL")

fig, ax = plt.subplots(1, 2, figsize=(10, 4))

ax[0].plot(V, pH)
ax[0].set_xlabel("Volume of base (mL)")
ax[0].set_ylabel("pH")
ax[0].set_title("Titration Curve")

ax[1].plot(V, slope)
ax[1].set_xlabel("Volume of base (mL)")
ax[1].set_ylabel("dpH/dV")
ax[1].set_title("Slope")

plt.tight_layout()
plt.savefig("titration.png")