import sys
import numpy as np

def run_aco_core(N, e, l, d, c, alpha, beta, evaporation_rate=0.1, max_no_improve=50):
    
    A = (e + d)[:, None] + c <= l
    pheromone = np.ones((N + 1, N + 1))
    Q = 1000.0
    
    total_cost = float('inf')
    best_route = None
    m = min(30, N)
    
    not_improved = 0
    
    while not_improved < max_no_improve:
        not_improved += 1
        for ants in range(m):
            visited = np.zeros(N + 1, dtype=bool)
            visited[0] = 1
            route = [0]
            curr = 0
            curr_time = 0
            L = 0

            for _ in range(N):
                feasible_nodes = np.flatnonzero(A[curr, :] & ~visited)
                arrival = curr_time + d[curr] + c[curr, feasible_nodes]
                
                valid_mask = arrival <= l[feasible_nodes]
                feasible_nodes = feasible_nodes[valid_mask]
                arrival = arrival[valid_mask]

                if len(feasible_nodes) == 0:
                    break

                start = np.maximum(arrival, e[feasible_nodes])
                finish = start + d[feasible_nodes]
                wait = start - arrival
                
                slack = np.maximum(0, l[feasible_nodes] - start)

                H = wait + slack + c[curr, feasible_nodes]
                
                H = np.maximum(H, 0)
                eta = 1 / (H + 1e-6)

                p_array = (pheromone[curr, feasible_nodes] ** alpha) * (eta ** beta)
                
                p_array = np.nan_to_num(p_array, nan=0.0, posinf=0.0, neginf=0.0)
                
                total_p = np.sum(p_array)
                
                if total_p <= 0:
                     idx = np.random.choice(len(feasible_nodes))
                else:
                     p_array = p_array / total_p
                     idx = np.random.choice(len(feasible_nodes), p=p_array)

                nxt = feasible_nodes[idx]
                s = start[idx]
                cost = c[curr, nxt]

                curr_time = s
                curr = nxt
                L += cost
                route.append(nxt)
                visited[nxt] = 1
            else:
                L += c[curr][0]
                
                route_np = np.array(route)
                from_nodes = route_np[:-1]
                to_nodes = route_np[1:]
                
                pheromone[from_nodes, to_nodes] += Q / L
                pheromone[to_nodes, from_nodes] += Q / L
                
                pheromone[curr][0] += Q / L
                pheromone[0][curr] += Q / L

                if L < total_cost:
                    best_route = route[1:]
                    total_cost = L
                    not_improved = 0

        pheromone *= (1 - evaporation_rate)

    return total_cost, best_route

def solve_vrptw_auto_tuned():
        
    N = int(input())
    e=[0]
    l=[10**9]
    d=[0]
    for i in range(N):
        e_i,l_i,d_i=map(int,input().split())
        e.append(e_i)
        l.append(l_i)
        d.append(d_i)
    e=np.array(e)
    l=np.array(l)
    d=np.array(d)
    c_list = [list(map(int,input().split())) for i in range(N+1)]
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
        cost, _ = run_aco_core(
            N, e, l, d, c, 
            alpha=params['alpha'], 
            beta=params['beta'], 
            max_no_improve=2
        )
        if cost < best_test_cost:
            best_test_cost = cost
            best_alpha = params['alpha']
            best_beta = params['beta']
            
    final_cost, final_route = run_aco_core(
        N, e, l, d, c, 
        alpha=best_alpha, 
        beta=best_beta, 
        max_no_improve=50
    )
    
    print(N)
    for i in final_route:
        print(i, end=' ')
    print()
solve_vrptw_auto_tuned()