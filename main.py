"""
Based on https://python.quantecon.org/optgrowth.html

which is © Copyright 2020, Thomas J. Sargent and John Stachurski.

This file is © Copyright 2020 by James Yu, modified and derived from the original, under Creative Commons Attribution-ShareAlike 4.0 International

https://creativecommons.org/licenses/by-sa/4.0/


A Python-based approximation of the infinite-horizon cake-eating problem.

y is used instead of W since I intend to use this more generally later on

"""

import numpy as np
import matplotlib.pyplot as plt

from model import OptimalGrowthModel

og = OptimalGrowthModel(u=np.log, β = 1)

v_greedy, v_solution = og.solve_model()

def v_star(grid, β):
  B = 1/(1-β)
  A = B*(np.log(1-β)+(β/(1-β))*np.log(β))
  return A + B*np.log(grid)

def σ_star(grid, β):
  return grid*(1-β)

fig, ax = plt.subplots()

ax.plot(og.grid, v_greedy, lw=2, alpha=0.6, label='approximate policy function')

ax.plot(og.grid, σ_star(og.grid, og.β), '--', lw=2, alpha=0.6, label='true policy function')
ax.legend()
plt.show()
