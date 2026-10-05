import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton, minimize

K = 10

def imbalance(x):
    return (2*x)**2/((1-x)*(1-x))-K

x_newton = newton(imbalance, 0.5)

result = minimize(lambda x: imbalance(x[0])**2, 0.5,
                  method="SLSQP", bounds=[(0, 0.999)])

x_slsqp = result.x[0]

print("Newton x:", x_newton)
print("SLSQP x:", x_slsqp)

H2 = 1-x_newton
I2 = 1-x_newton
HI = 2*x_newton

print("H2:", H2)
print("I2:", I2)
print("HI:", HI)

x = np.linspace(0, 0.99, 200)

H2_values = 1-x
I2_values = 1-x
HI_values = 2*x

plt.figure()
plt.plot(x, H2_values, label="H2")
plt.plot(x, I2_values, label="I2")
plt.plot(x, HI_values, label="HI")
plt.axvline(x_newton, linestyle="--")
plt.xlabel("Extent x")
plt.ylabel("Amount (mol)")
plt.legend()
plt.tight_layout()
plt.savefig("equilibrium.png")