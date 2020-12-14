import numpy as np
from scipy.interpolate import interp1d 
# linear interpolation is a type of approximation which tries to fit linear polynomials to a dataset
from scipy.optimize import minimize_scalar
# this is just a minimization function

class OptimalGrowthModel:
  # this class fits a solution for our value function to a series of values and parameters

  # replace β = 0.8 with a particular value; this is the amount by which we discount the future
  def __init__(self, u, β = 0.8, grid_max=4, grid_size=120):
    self.u, self.β, = u, β
    self.grid = np.linspace(1e-4, grid_max, grid_size)
    # this generates a vector from 0.0001 to 4 with 120 entries

  def state_action_value(self, c, y, v_array):
    v = interp1d(self.grid, v_array, fill_value="extrapolate")
    # compared to the original, I had to add an extrapolated fill due to an out of bounds error
    return self.u(c) + self.β*v(y-c)
    # this is the Bellman equation:
    # V(W) = u(c)+V(W-c) after we substitute the constraint

  def T(self, v):
    # this function applies the value function across the grid vector
    # in order to find an optimal solution given the existing vector
    v_new = np.empty_like(v)
    v_greedy = np.empty_like(v)

    for i in range(len(self.grid)):
      y = self.grid[i]
      c_star, v_max = self.maximize(self.state_action_value, 1e-10, y, (y, v))
      v_new[i] = v_max
      v_greedy[i] = c_star

    return v_greedy, v_new

  def maximize(self, g, a, b, args):
    # as noted in the original document, we can minimize the negative to maximize the original function
    result = minimize_scalar(lambda x: -g(x, *args), bounds=(a, b), method='bounded')
    return result.x, -result.fun # return optimal x and value

  def solve_model(self):
    v = self.u(self.grid)

    tol=1e-4 # how precise we want the answer to be
    max_iter=1000 # if we iterate more than this many times, we failed
    i = 0
    error = tol + 1

    while i < max_iter and error > tol:
      # while we haven't converged, continue iterating the value function
      # this is a method of solving dynamic programs known as value function iteration
      v_greedy, v_new = self.T(v)
      error = np.max(np.abs(v - v_new))
      i += 1
      v = v_new

      if i % 10 == 0:
        print(f"Error at iteration {i} is {error}.")

    if i == max_iter:
      print("Failed to converge!")
    else:
      print(f"\nConverged in {i} iterations.")

    return v_greedy, v_new