import numpy as np
from scipy.optimize import newton, minimize

def f(x):
    return (x-3)**2+1

def df(x):
    return 2*(x-3)

def ddf(x):
    return 2

def g(x):
    return x**4-3*x**2+x+5

def dg(x):
    return 4*x**3-6*x+1

def ddg(x):
    return 12*x**2-6

# Gradient descent
x = 0
alpha = 0.1

for i in range(100):
    x = x-alpha*df(x)

print("Gradient descent:", x)

# Newton
x = newton(df, 0, fprime=ddf)
print("Newton:", x)

# SLSQP
result = minimize(f, 0, method="SLSQP")
print("SLSQP:", result.x[0])

x = newton(dg, 0, fprime=ddg)
print("Newton from 0:", x)
print("Second derivative:", ddg(x))

x = newton(dg, 2, fprime=ddg)
print("Newton from 2:", x)
print("Second derivative:", ddg(x))

result = minimize(g, 0, method="SLSQP")
print("SLSQP from 0:", result.x[0])

result = minimize(g, 2, method="SLSQP")
print("SLSQP from 2:", result.x[0])

x = 0
for i in range(1000):
    x = x-0.01*dg(x)
print("Gradient descent from 0:", x)

x = 2
for i in range(1000):
    x = x-0.01*dg(x)
print("Gradient descent from 2:", x)