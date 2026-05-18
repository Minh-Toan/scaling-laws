# Project Overview

This repository contains code and data for theoretical predictions, simulations, and experiments for

**Sharp feature-learning transitions and Bayes-optimal neural scaling laws in extensive-width networks**

(https://arxiv.org/pdf/2605.10395)

---

# Core Files

- **func.py**  
  Implements theoretical predictions.

- **hmc.py**  
  Implements Hamiltonian Monte-Carlo (HMC) simulations.

- **sgd.py**  
  Implements Stochastic Gradient Descent (SGD) with the Adam optimizer.

---

# Data Generation Scripts

These scripts generate datasets used in the experiments and figures:

- **data_err.py**  
  Generates data in `err_data_beta_03/`

- **data_overlap.py**  
  Generates data in `overlaps_data/`

- **data_sgd_varying_widths_1.py**  
  Generates data in `sgd_data_beta_1/`

- **data_sgd_varying_widths_2.py**  
  Generates data in `sgd_data_beta_1_new/`

---

# Data Folders

- **err_data_beta_03/**  
  Used for Figure 2 (left panel)

- **overlaps_data/**  
  Used for overlap vs. feature index plots

- **sgd_data_beta_1/**  
  Used for Figure 1 (middle panel)

- **sgd_data_beta_1_new/**  
  Used for Figure 1 (left panel), particularly scaling laws in data

- **sgd_data_beta_1_kc/**  
  Contains generalization error results when student width is equal to the effective width ($k_c$), $\beta=1$

---

# Data Processing and Plotting

All plotting and analysis notebooks are located as: **plot_fig_*.ipynb**  Note that **plot_fig_2b.ipynb**  can also generate overlap vs. feature index plots used in the appendix.
