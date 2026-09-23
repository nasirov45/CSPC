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
