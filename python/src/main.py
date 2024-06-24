from ExtremalPoints import ExtremalPoint
import numpy as np

#This script implements the algorithm 1 for a wave equation with point sources

# Ideas: Warmstart with equidistant kinks and jumps

d = 2

def solve_linear_problem(active_set: list[ExtremalPoint], problem: Type[ControlProblem.]:
    return 1, 2
def main():
    # Set all problem parameters, active set, 
    N = 1
    d = 2
    active_set = [ExtremalPoint([np.sqrt(1/2), np.sqrt(1/2)], .2, 0)]
    print(active_set[0].sigma)
    weights, ell = solve_linear_problem(active_set)
    
    

if __name__ == "__main__":
    main()