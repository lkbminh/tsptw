import sys
import numpy as np

def ACO(N, e, l, d, c, alpha=1, beta=1, rho=0.1, Q=1000, max_no_improvements=100):
    e = np.array(e)
    l = np.array(l)
    d = np.array(d)
    c = np.array(c)

    A = (e + d)[:, None] + c <= l
    pheromone = np.ones((N + 1, N + 1))

    best_route = None
    best_cost = float('inf')
    
    m = min(50, N)
    not_improved = 0

    while not_improved < max_no_improvements:
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
                slack = np.maximum(0, l[feasible_nodes] - finish)

                H = np.maximum(0, wait + slack + c[i, feasible_nodes])
                eta = 1 / (H + 1e-6)

                pheromone_level = pheromone[i, feasible_nodes] ** alpha
                heuristic_level = eta ** beta
                
                p_num = pheromone_level * heuristic_level
                p_num = np.nan_to_num(p_num, nan=0.0, posinf=0.0, neginf=0.0)
                p_den = np.sum(p_num)
                
                if p_den <= 0:
                    idx = np.random.choice(len(feasible_nodes))
                else:
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
                pheromone[to_nodes, from_nodes] += Q / L

                if L < best_cost:
                    best_route = route[1:]
                    best_cost = L
                    not_improved = 0

        pheromone *= (1 - rho)
        
        if best_route is not None:
            elite_route = np.concatenate(([0], best_route, [0]))
            e_from = elite_route[:-1]
            e_to = elite_route[1:]
            pheromone[e_from, e_to] += Q / best_cost
            pheromone[e_to, e_from] += Q / best_cost

    return best_route, best_cost

def solve_vrptw_auto_tuned():
    input_data = sys.stdin.read().split()
    if not input_data:
        return
        
    N = int(input_data[0])
    
    e_list = [0]
    l_list = [10**9]
    d_list = [0]
    
    for i in range(1, 1 + N * 3, 3):
        e_list.append(int(input_data[i]))
        l_list.append(int(input_data[i+1]))
        d_list.append(int(input_data[i+2]))
        
    e = np.array(e_list)
    l = np.array(l_list)
    d = np.array(d_list)
    
    idx = 1 + N * 3
    c_list = []
    for _ in range(N + 1):
        c_list.append([int(input_data[i]) for i in range(idx, idx + N + 1)])
        idx += (N + 1)
    c = np.array(c_list)
    
    l[0] = max(l[i] + d[i] + c[i][0] for i in range(1, N + 1))

    candidate_params = [
        {'alpha': 1.0, 'beta': 2.0},
        {'alpha': 1.0, 'beta': 5.0},
        {'alpha': 2.0, 'beta': 1.0},
        {'alpha': 1.5, 'beta': 3.0}
    ]
    
    best_alpha = 1.0
    best_beta = 2.0
    best_test_cost = float('inf')
    
    for params in candidate_params:
        cost, _ = ACO(N, e, l, d, c, alpha=params['alpha'], beta=params['beta'], max_no_improvements=2)
        if cost < best_test_cost:
            best_test_cost = cost
            best_alpha = params['alpha']
            best_beta = params['beta']
            
    final_cost, final_route = ACO(N, e, l, d, c, alpha=best_alpha, beta=best_beta, max_no_improvements=50)
    
    print(N)
    for i in final_route:
        print(i, end=' ')
    print()

if __name__ == "__main__":
    solve_vrptw_auto_tuned()
