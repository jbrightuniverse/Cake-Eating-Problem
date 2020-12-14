"""
Based on https://python.quantecon.org/optgrowth.html

which is © Copyright 2020, Thomas J. Sargent and John Stachurski.

This file is © Copyright 2020 by James Yu, modified and derived from the original, under Creative Commons Attribution-ShareAlike 4.0 International

https://creativecommons.org/licenses/by-sa/4.0/


A Python-based approximation of the infinite-horizon cake-eating problem.

y is used instead of W since I intend to use this more generally later on

Main changes:
- adjusted all equations to use the basic Cake-Eating optimal consumption plan problem (logarithmic utility, W' = W-c)
- merged value function solver, Bellman operator and general maximizer into OptimalGrowthModel class
- adjusted plotter to plot both policy function and value function at the same time
"""

import numpy as np
import matplotlib.pyplot as plt

from model import OptimalGrowthModel

og = OptimalGrowthModel(u=np.log, β = 0.8)

v_greedy, v_solution = og.solve_model()

# this is the actual solution for the value function
# derived from Prof. Michal Skzup's notes on Dynamic Programming
# for UBCV Econ 305
def v_star(grid, β):
  B = 1/(1-β)
  A = B*(np.log(1-β)+(β/(1-β))*np.log(β))
  return A + B*np.log(grid)

# similarly, this is the optimal policy function, derived from the same notes, which states the optimal consumption choice given a particular level of cake, material, or whatever
def σ_star(grid, β):
  return grid*(1-β)

fig, sub = plt.subplots(2, sharex=True)
fig.suptitle("Cake-Eating Problem")

sub[0].plot(og.grid, v_greedy, lw=2, alpha=0.5, label='appx policy function')
sub[0].plot(og.grid, σ_star(og.grid, og.β), '--', lw=2, alpha=0.5, label='true policy function')

sub[1].plot(og.grid, v_solution, lw=2, alpha=0.5, label='appx value function')
sub[1].plot(og.grid, v_star(og.grid, og.β), lw=2, alpha=0.5, label='True value function')

sub[0].legend()
sub[1].legend()

plt.show()