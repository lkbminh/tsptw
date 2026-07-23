# Traveling Salesperson Problem with Time Windows (TSPTW) Solvers

This repository contains a collection of Python scripts designed to solve a single-vehicle routing problem with time windows and service durations. The solution methods range from exact mathematical optimization to advanced metaheuristics.

## Algorithms Implemented

The following optimization approaches are provided in this collection:

* **Ant Colony Optimization (ACO):** A probabilistic technique using a pheromone matrix, evaporation rates (`rho`), and heuristic information, initialized with a greedy algorithm.


* **Constraint Programming (CP):** An exact approach utilizing the `cp_model` from Google's OR-Tools to enforce time constraints and Hamiltonian circuit requirements.


* **Local Search:** A heuristic method that starts with a greedy initial solution and iteratively attempts to improve the route by swapping node positions while maintaining time-window feasibility.


* **Adaptive Large Neighborhood Search (ALNS):** A metaheuristic implementing multiple destroy operators (random, segment, worst) and repair operators (greedy, regret-2, random) to dynamically explore the solution space, utilizing Simulated Annealing for acceptance criteria.


* **Branch and Bound:** An exact algorithmic implementation utilizing backtracking and lower-bound cost estimations to find the optimal route.


* **Mixed Integer Programming (MIP):** An exact solver utilizing the `pywraplp` module from Google's OR-Tools to interface with the SCIP solver, complete with subtour elimination and time variable constraints.



## Prerequisites

To run these scripts, you will need Python installed along with the following third-party libraries:

* `numpy` (Required for matrix operations in ACO and ALNS)


* `ortools` (Required for the CP and MIP solver implementations)



## Input Format

All scripts read the problem instance from standard input (`sys.stdin`). The expected format is as follows:

1. **First Line:** An integer `N` representing the number of target nodes (excluding the depot).


2. **Next `N` Lines:** Three space-separated integers per line representing the earliest start time (`e`), latest start time (`l`), and service duration (`d`) for each target node.


3. **Remaining Lines:** The cost or distance matrix `c`, containing `N + 1` rows and `N + 1` columns of integers, representing the travel cost between all nodes including the depot.


## Output Format

If a feasible or optimal route is found, the scripts will output:

1. The number of target nodes `N`.


2. A space-separated sequence of integers representing the visited node order, omitting the starting and ending depot.



If the exact solvers (CP or MIP) fail to find an optimal solution, they will print "No optimal solution".

## Usage

You can execute any of the scripts from the command line by passing a formatted text file into standard input.

```bash
python aco.py < input.txt
python alns.py < input.txt
python cp.py < input.txt
python greedy.py < input.txt
python local_search.py < input.txt
python branch_and_bound.py < input.txt
python mip.py < input.txt

```