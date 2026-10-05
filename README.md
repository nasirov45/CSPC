# CSPC - Computer Science for Physics and Chemistry

My coursework repository. Each practical is under PW<n>/Lab <X>/.

## Setup

Create the environment for a given lab:

```bash
conda env create -f PW<n>/Lab\ <X>/environment.yml
conda activate cspc
```

---

## PW1 - Lab A: Reproducible Foundations

**What I built:**

* Implemented pure-Python and NumPy versions of a radioactive decay simulation.
* Added tests for the initial atom count, negative decay rates, and the analytical decay law.

**Speed comparison (loop vs NumPy):**

* loop : 1.830334 s
* numpy : 0.000155 s
* speed-up: 11780.18 x faster

**Tests:** all passing? **Yes**

**Conclusion:**

* All three tests passed successfully.
* The NumPy implementation was much faster than the pure-Python loop for 200,000 atoms.
* I learned how vectorisation can significantly improve performance and how pytest can be used to test different parts of a simulation.

## PW1 --- Lab B

The observed decay data showed a decreasing number of counts over time, following the expected decay pattern.

The observed data matched the analytical decay law reasonably well, as the points followed a similar decreasing shape to the analytical curve.

The Snakemake pipeline takes the observed CSV data and `plot.py` as inputs and automatically generates `figure.png`, rebuilding it whenever an input file changes.


## PW2 — Lab A

### Motion from Tracking Data

The mean acceleration calculated from the free-fall data was **-8.58 m/s²**, which is close to the expected value of **-9.81 m/s²**.

The acceleration is much noisier than the position because differentiation amplifies small measurement errors and noise in the data.

After integrating the acceleration twice, the recovered position was close to the original position. The largest difference was approximately **0.785 m**, which is within the expected 1 metre difference.

### Bonus — 2D Trajectory

For the 2D trajectory data, the x and y coordinates were differentiated separately using `np.gradient` to calculate the velocity in each direction. The speed was then calculated from the x and y velocity components and plotted against time.


## PW2 — Lab B

### Optimisation in Chemistry

Three optimisation methods were compared: gradient descent, Newton's method, and SLSQP. For the simple convex function, all three methods converged to the same minimum at approximately **x = 3**.

For the more complicated function, the methods did not always give the same result. The starting point affected the result because the function has multiple stationary points. Newton's method can converge to a maximum as well as a minimum, so the second derivative must be checked. This shows that both the starting point and the optimisation algorithm matter for a more complicated landscape.

### Reaction Rate

The first-order reaction data was fitted using SLSQP. The fitted rate constant **k** was close to **0.25**, and the fitted exponential curve followed the measured concentration data.

### Chemical Equilibrium

For the reaction H₂ + I₂ ⇌ 2HI, the equilibrium extent was found using both Newton's method and SLSQP. Both methods gave approximately the same equilibrium value. The equilibrium composition was calculated from:

* H₂ = 1 − x mol
* I₂ = 1 − x mol
* HI = 2x mol

### Titration Equivalence Point

The pH slope was calculated using `np.gradient`. The equivalence point was found by locating the volume where the slope was largest. The equivalence point was approximately **50 mL**, where the pH curve changes most rapidly.
