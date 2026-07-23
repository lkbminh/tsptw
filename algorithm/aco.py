import numpy as np
import sys

def ACO(N, e, l, d, c, alpha=1, beta=2, rho=0.1, Q=1.0, max_no_improvements=50):
    e, l, d, c = map(np.array, (e, l, d, c))

    temp_route, temp_cost = greedy(N, e, l, d, c)
    if not temp_route:
        best_route = None
        best_cost = float('inf')
    else:
        best_route = temp_route
        best_cost = temp_cost

    if best_cost == float('inf'):
        tau_max, tau_min = 1.0, 0.01 
    else:
        tau_max, tau_min = update_bounds(best_cost, rho, N)

    A = (e + d)[:, None] + c <= l
    pheromone = np.full((N + 1, N + 1), tau_min)
    np.fill_diagonal(pheromone, 0.0)

    if best_route is not None and best_cost < float('inf'):
        r = np.concatenate(([0], best_route, [0]))
        pheromone[r[:-1], r[1:]] = tau_max 

    m = min(50, N)
    not_improved = 0
    iterations = 0

    while not_improved < max_no_improvements:
        iterations += 1
        not_improved += 1
        iter_best_route, iter_best_cost = None, float('inf')
        dead_ants = 0
        total_ants = 0

        for _ in range(m):
            total_ants += 1
            visited = np.zeros(N + 1, dtype=bool)
            visited[0] = True
            route, i, time, L = [0], 0, 0.0, 0.0

            for _ in range(N):
                feasible_nodes = np.flatnonzero(A[i] & ~visited)
                arrival = time + d[i] + c[i, feasible_nodes]
                valid = arrival <= l[feasible_nodes]
                feasible_nodes, arrival = feasible_nodes[valid], arrival[valid]

                if len(feasible_nodes) == 0:
                    dead_ants += 1
                    break

                start = np.maximum(arrival, e[feasible_nodes])
                wait = start - arrival
                slack = l[feasible_nodes] - arrival
                H = wait + slack + c[i, feasible_nodes]
                eta = 1.0 / (H + 1e-6)

                p_num = pheromone[i, feasible_nodes] ** alpha * eta ** beta
                p_num = np.nan_to_num(p_num, nan=0.0, posinf=0.0, neginf=0.0)
                p_den = p_num.sum()

                if p_den <= 0:
                    idx = np.random.randint(len(feasible_nodes))
                else:
                    p = p_num / p_den
                    p = p / p.sum() 
                    idx = np.random.choice(len(feasible_nodes), p=p)

                j = feasible_nodes[idx]
                time = start[idx]
                L += c[i, j]
                route.append(j)
                visited[j] = True
                i = j

            else:
                L += c[i, 0]
                route.append(0)
                r = np.array(route)

                if L < iter_best_cost:
                    iter_best_route = r
                    iter_best_cost = L

                if L < best_cost:
                    best_route = r[1:-1]
                    best_cost = L
                    not_improved = 0
                    tau_max, tau_min = update_bounds(best_cost, rho, N)  

        pheromone *= (1 - rho)                                           
        
        # Elitist deposit using Q
        if iter_best_route is not None:
            pheromone[iter_best_route[:-1], iter_best_route[1:]] += Q / iter_best_cost

        pheromone = np.clip(pheromone, tau_min, tau_max)

    return best_route, best_cost    

def calculate_bf(pheromone, threshold=0.5):
    p = pheromone.copy()
    np.fill_diagonal(p, 0.0)
    row_max = p.max(axis=1, keepdims=True)
    valid_rows = row_max.squeeze() > 0
    active = (p[valid_rows] >= threshold * row_max[valid_rows]).sum(axis=1)
    return float(active.mean()) if active.size > 0 else 1.0

def greedy(N, e, l, d, c):
    visited = [0] * (N + 1)
    route = []
    time = 0
    i = 0
    cost = 0

    while len(route) < N:
        candidates = []
        for j in range(1, N + 1):
            if not visited[j]:
                arrival = time + d[i] + c[i][j]
                
                if arrival <= l[j]:
                    start = max(arrival, e[j])
                    finish = start + d[j]
                    wait = start - arrival
                    slack = l[j] - arrival

                    H = c[i][j] + wait + 0.35 * slack  
                    candidates.append((H, j, start, finish))
            
        if not candidates:
            return [], float('inf')
        
        h, j, s, f = min(candidates, key= lambda x: x[0])

        time = s
        cost += c[i][j]
        i = j

        route.append(j)
        visited[j] = 1

    cost += c[i][0]

    return route, cost


def update_bounds(best_cost, rho, N, Q = 1.0):
    tau_max = Q / (rho * best_cost)
    tau_min = (1/N) * tau_max

    return tau_max, tau_min

def main():
    inp = sys.stdin.read().strip().splitlines()
    if not inp:
        return

    N = int(inp[0])
    nodes = [i for i in range(N + 1)]
    e = [0]      
    l = [99999]  
    d = [0]
    c = []

    for i in range(N):
        e_i, l_i, d_i = inp[i + 1].split()
        e.append(int(e_i))
        l.append(int(l_i))
        d.append(int(d_i))

    for i in nodes:
        c.append([int(x) for x in inp[i + N + 1].split()])
    
    l[0] = max(l[i] + d[i] + c[i][0] for i in range(1, N + 1))

    route, cost = ACO(N, e, l, d, c, alpha=1, beta=2, rho=0.1, max_no_improvements=100)
    
    print(N)
    if route is not None:
        for i in route:
            print(i, end=' ')
        print()

if __name__ == "__main__":
    main()