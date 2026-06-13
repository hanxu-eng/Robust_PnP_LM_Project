# Project Development Constraints

- Do not call OpenCV `solvePnP` or related ready-made PnP solvers.
- Do not call `scipy.optimize.least_squares` or similar ready-made nonlinear least-squares solvers.
- All experiment figures must be saved as vector PDF files.
- Experiment results, tables, and figures must come from actual code execution.
- `quick` mode is for fast validation; `full` mode is for formal experiments.
- Prioritize runnable, reproducible, and readable code.
- All random experiments must use fixed seeds.
- Keep the implementation focused on NumPy and Matplotlib plus Python standard-library modules.
