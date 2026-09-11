# CSPC
Computer Science Programming Course(cspc)

## PW1 — Lab A

### Tests
All three tests pass successfully with `pytest -v`.

- Test for initial atom count: passed
- Test for negative decay rate: passed
- Test comparing the simulation average with the analytical law: passed

### Speed comparison

The simulation was tested with 200,000 atoms.

- Pure-Python loop: **1.830334 seconds**
- NumPy version: **0.000155 seconds**
- NumPy speed-up: **11,780.18× faster**

### Conclusion

The NumPy vectorised implementation is significantly faster than the pure-Python loop while producing the same type of decay simulation. The speed comparison shows that NumPy is much more efficient for large numbers of atoms.