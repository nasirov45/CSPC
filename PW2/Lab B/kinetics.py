import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

data = np.loadtxt("kinetics.csv", delimiter=",", skiprows=1)

t = data[:, 0]
C = data[:, 1]

C0 = C[0]

def error(k):
    predicted = C0*np.exp(-k*t)
    return np.sum((C-predicted)**2)

result = minimize(error, 0.5, method="SLSQP", bounds=[(0, 5)])

k = result.x[0]

print("Fitted k:", k)

fitted = C0*np.exp(-k*t)

plt.figure()
plt.scatter(t, C, label="Measured")
plt.plot(t, fitted, label="Fitted")
plt.xlabel("Time")
plt.ylabel("Concentration")
plt.legend()
plt.tight_layout()
plt.savefig("kinetics.png")