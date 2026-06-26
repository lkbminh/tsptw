import sys
import numpy as np
import random
from ortools.sat.python import cp_model
from ortools.linear_solver import pywraplp
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

def cp(nodes, e, l, d, c):
    model = cp_model.CpModel()

    x = {}
    t = {}
    arcs = []
    
    # We will keep a separate list of valid pairs to prevent looping over 
    # impossible arcs later in the constraints and objective function.
    valid_pairs = []
    
    # 1. Safer Upper Bound for Return Time
    # sum(l) can create a massive integer domain. A tighter, memory-safe bound:
    max_horizon = max(l) + sum(max(row) for row in c)
    l[0] = max_horizon
    t_return = model.new_int_var(0, max_horizon, 't_return')

    for i in nodes:
        t[i] = model.new_int_var(e[i], l[i], f't[{i}]')

    # 2. Graph Sparsification (The MLE Fix)
    for i in nodes:
        for j in nodes:
            if i == j:
                continue
            
            # PRUNING: If leaving `i` at the earliest possible time still 
            # makes us arrive late for `j`, this path is physically impossible.
            if j != 0 and e[i] + d[i] + c[i][j] > l[j]:
                continue
                
            # We only create variables for temporally possible arcs
            x[i, j] = model.new_bool_var(f'x[{i}][{j}]')
            arcs.append((i, j, x[i, j]))
            valid_pairs.append((i, j))

    # This single line forces a valid, unbroken Hamiltonian cycle
    model.add_circuit(arcs)

    # 3. Only build Time Constraints for valid pairs
    for i, j in valid_pairs:
        if j != 0:
            model.add(t[j] >= t[i] + d[i] + c[i][j]).only_enforce_if(x[i, j])
        else:
            model.add(t_return >= t[i] + d[i] + c[i][j]).only_enforce_if(x[i, j])

    # 4. Clean Objective Function
    # By iterating only over valid_pairs, we prevent Python from generating 
    # a million-element linear expression in memory.
    total_cost = sum(c[i][j] * x[i, j] for i, j in valid_pairs)
    model.minimize(total_cost)

    solver = cp_model.CpSolver()
    
    # Optional: If you want to explicitly cap the solver's CPU time to prevent hanging
    # solver.parameters.max_time_in_seconds = 300 
    
    status = solver.solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        cost = solver.objective_value
        route = []
        current_node = 0

        while True:
            for j in nodes:
                # We must check if (current_node, j) actually exists in x 
                # because we skipped creating the impossible ones!
                if (current_node, j) in x and solver.boolean_value(x[current_node, j]):
                    if j != 0:
                        route.append(j)
                    current_node = j
                    break
            if current_node == 0:
                break
        return cost, route
        
    else:
        return None, None

def mip(nodes, e, l, d, c):
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        return None, None
    
    N = len(nodes) - 1
    x = {}
    t = {}
    u = {}
    
    outgoing_valid = {i: [] for i in nodes}
    incoming_valid = {j: [] for j in nodes}
    valid_pairs = []

    for i in nodes:
        t[i] = solver.IntVar(e[i], l[i], f't[{i}]')
    
    max_horizon = max(l) + sum(max(row) for row in c)
    t_return = solver.IntVar(0, max_horizon, 't_return')

    for i in nodes:
        if i != 0:
            u[i] = solver.IntVar(1, N, f'u[{i}]')

    # 3. Graph Sparsification (FIXED)
    for i in nodes:
        for j in nodes:
            if i == j:
                continue
                
            # PRUNING: We MUST exempt j=0 from this check, otherwise we delete 
            # all roads returning to the depot!
            if j != 0 and e[i] + d[i] + c[i][j] > l[j]:
                continue

            x[i, j] = solver.IntVar(0, 1, f'x[{i}][{j}]')
            outgoing_valid[i].append(j)
            incoming_valid[j].append(i)
            valid_pairs.append((i, j))

    # 4. Optimized Routing Constraints
    for i in nodes:
        if not outgoing_valid[i]:
            return None, None
        solver.Add(sum(x[i, j] for j in outgoing_valid[i]) == 1)

    for j in nodes:
        if not incoming_valid[j]:
            return None, None
        solver.Add(sum(x[i, j] for i in incoming_valid[j]) == 1)

    # 5. Add Constraints ONLY for valid pairs
    for i, j in valid_pairs:
        M = max(0, l[i] - e[j] + d[i] + c[i][j])
        
        if j != 0:
            solver.Add(t[j] >= t[i] + d[i] + c[i][j] - M * (1 - x[i, j]))
        else:
            solver.Add(t_return >= t[i] + d[i] + c[i][j] - M * (1 - x[i, j]))

        if i != 0 and j != 0:
            solver.Add(u[i] - u[j] + N * x[i, j] <= N - 1)

    solver.Minimize(sum(x[i, j] * c[i][j] for i, j in valid_pairs))

    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        cost = solver.Objective().Value()
        route = []
        current_node = 0

        while True:
            # Added a safety toggle to prevent infinite loops if SCIP returns fractions
            found_next = False 
            for j in outgoing_valid[current_node]:
                if x[current_node, j].solution_value() > 0.5:
                    if j != 0:
                        route.append(j)
                    current_node = j
                    found_next = True
                    break
            
            if current_node == 0 or not found_next:
                break
                
        return cost, route
        
    else:
        return None, None

def greedy(N, e, l, d, c):
    def build(mode):
        # 0 = min start time, 1 = min finish time
        # Using a list guarantees deterministic tie-breaking
        unvisited = list(range(1, N + 1))
        route = []
        t, i, cost = 0, 0, 0
        
        while unvisited:
            candidates = []
            for j in unvisited:
                # arrival = start time at current node + duration + travel time
                arrival = t + d[i] + c[i][j] 
                
                if arrival <= l[j]:
                    start = max(arrival, e[j])
                    finish = start + d[j]
                    candidates.append((start, finish, c[i][j], j))
                    
            if not candidates:
                return None, float('inf')
            
            best_start, best_finish, best_cost, best_node = min(candidates, key = lambda x: x[mode])
            
            t = best_start
            i = best_node
            cost += best_cost

            route.append(best_node)
            unvisited.remove(best_node)
        
        return route, cost
    
    start_route, start_cost = build(0)
    finish_route, finish_cost = build(1)

    if start_route is None and finish_route is None:
        return None, float('inf')

    # The float('inf') ensures a failed route naturally loses this check
    if start_cost < finish_cost:
        return start_route, start_cost
    else:
        return finish_route, finish_cost
            
def feasible(route, e, l, d, c):
    curr_time = 0
    curr_cost = 0
    curr_node = 0

    for node in route:
        arrival = curr_time + c[curr_node][node]
        if arrival > l[node]:
            return False, float('inf')
        start = max(arrival, e[node])
        curr_time = start + d[node]
        curr_cost += c[curr_node][node]
        curr_node = node
    
    curr_cost += c[curr_node][0]
    return True, curr_cost

def local_search(nodes, e, l, d, c, mode=3):
    best_cost, best_route = greedy(nodes, e, l, d, c, mode)
    N = len(best_route) - 1
    still_improve = True

    while still_improve:
        still_improve = False

        for i in range(0, N):
            for j in range(i, N):
                if i != j:
                    new_route = best_route.copy()
                    new_route[i], new_route[j] = new_route[j], new_route[i]
                    feasible, new_cost = feasible(new_route, e, l, d, c)
                    if feasible and new_cost < best_cost:
                        best_route = new_route
                        best_cost = new_cost
                        still_improve = True
                        break
            if still_improve:
                break

    return best_cost, best_route

def ACO(N, e, l, d, c, alpha=1, beta=1, evaporation_rate=0.5):
    e = np.array(e)
    l = np.array(l)
    d = np.array(d)
    c = np.array(c)

    pheromone = np.ones((N + 1, N + 1))
    Q = 100
    A = (e + d)[:, None] + c <= l
    
    total_cost = float('inf')
    nodes = list(range(1, N + 1))
    m = min(50, N)
    best_route = None
    not_improved = 0

    while not_improved < N * 10:
        not_improved += 1
        
        for ants in range(m):
            visited = np.zeros(N + 1, dtype=bool)
            visited[0] = True
            
            route = [0]
            i = 0
            time = 0
            L = 0

            for _ in range(N):
                feasible_nodes = np.flatnonzero(A[i, :] & ~visited)
                arrival = time + d[i] + c[i, feasible_nodes]
                
                valid_mask = arrival <= l[feasible_nodes]
                feasible_nodes = feasible_nodes[valid_mask]
                arrival = arrival[valid_mask]

                if len(feasible_nodes) == 0:
                    break

                start = np.maximum(arrival, e[feasible_nodes])
                finish = start + d[feasible_nodes]
                wait = start - arrival
                slack = l[feasible_nodes] - finish

                H = wait + slack + c[i, feasible_nodes]
                eta = 1 / (H + 1e-6)

                pheromone_level = pheromone[i, feasible_nodes] ** alpha
                heuristic_level = eta ** beta
                
                p_num = pheromone_level * heuristic_level
                p_den = np.sum(p_num)
                p = p_num / p_den
                
                idx = np.random.choice(len(feasible_nodes), p=p)
                
                j = feasible_nodes[idx]
                s = start[idx]
                cost = c[i, j]

                time = s
                i = j
                L += cost
                
                route.append(j)
                visited[j] = True
                
            else:
                L += c[i, 0]
                route = np.array(route)
                from_nodes = route[:-1]
                to_nodes = route[1:]
                
                pheromone[from_nodes, to_nodes] += Q / L

                if L < total_cost:
                    best_route = route[1:]
                    total_cost = L
                    not_improved = 0

        pheromone *= (1 - evaporation_rate)


def print_sol(route):
    print(len(route))
    for i in route:
        print(i, end = ' ')
    print()

def main():
    inp = sys.stdin.read().strip().splitlines()

    N = int(inp[0])
    nodes = [i for i in range(N + 1)]
    e = [0]      # Node 0 is the depot, so its earliest time is 0
    l = [99999]  # Node 0 is the depot, so its latest time can be set to infinity
    d = [0]      # Node 0 is the depot, so its service time is 0
    c = []

    for i in range(N):
        e_i, l_i, d_i = inp[i + 1].split()
        e.append(int(e_i))
        l.append(int(l_i))
        d.append(int(d_i))

    l[0] = max(l[i] + d[i] + c[i][0] for i in range(1, N + 1))

    for i in nodes:
        c.append([int(x) for x in inp[i + N + 1].split()])

    ACO(N, e, l, d, c)
if __name__ == "__main__":
    main()