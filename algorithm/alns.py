import numpy as np
import sys
import math

def ALNS(N, e, l, d, c, max_no_improvements=300):
    # Convert all inputs to NumPy arrays for C-level speed
    e, l, d, c = map(np.array, (e, l, d, c))
    
    def evaluate_route(route):
        if not route:
            return float('inf'), False
        
        route = np.concatenate(([0], route, [0]))
        current_time = 0.0
        total_violation = 0.0
        total_travel = 0.0
        
        for i in range(len(route) - 1):
            u, v = route[i], route[i+1]
            arr_time = current_time + d[u] + c[u, v]
            start_time = max(arr_time, e[v])
            
            total_violation += max(0.0, arr_time - l[v])
            total_travel += c[u, v]
            current_time = start_time
            
        is_feasible = (total_violation == 0)
        
        penalized_cost = total_travel + (total_violation * 1000)
            
        return penalized_cost, is_feasible
    
    def destroy_random(route, q):
        removed = np.random.choice(route, q, replace=False).tolist()
        new_route = [x for x in route if x not in removed]
        return new_route, removed

    def destroy_segment(route, q):
        start_idx = np.random.randint(0, max(1, len(route) - q + 1))
        removed = route[start_idx : start_idx + q]
        new_route = route[:start_idx] + route[start_idx + q:]
        return new_route, removed

    def destroy_worst(route, q):
        if not route: return [], []
        
        prev_n = np.array([0] + route[:-1])
        curr_n = np.array(route)
        next_n = np.array(route[1:] + [0])
        
        contributions = c[prev_n, curr_n] + c[curr_n, next_n] - c[prev_n, next_n]
        
        worst_indices = np.argsort(contributions)[-q:]
        worst_nodes = set(curr_n[worst_indices])
        
        removed = [x for x in route if x in worst_nodes]
        new_route = [x for x in route if x not in worst_nodes]
        return new_route, removed

    def repair_greedy(route, unassigned):
        for node in unassigned:
            prev_n = np.array([0] + route)
            next_n = np.array(route + [0])
            
            # 1. Calculate current arrival times at each node in prev_n
            times_prev = np.zeros(len(prev_n))
            curr_time = 0.0
            for i in range(1, len(prev_n)):
                p = prev_n[i-1]
                curr = prev_n[i]
                curr_time = max(curr_time, e[p]) + d[p] + c[p, curr]
                times_prev[i] = curr_time
                
            arr_node = np.maximum(times_prev, e[prev_n]) + d[prev_n] + c[prev_n, node]
            viol_node = np.maximum(0, arr_node - l[node])
            
            start_node = np.maximum(arr_node, e[node])
            arr_next = start_node + d[node] + c[node, next_n]
            viol_next = np.maximum(0, arr_next - l[next_n])
            
            time_penalty = (viol_node + viol_next) * 1000
            
            deltas = c[prev_n, node] + c[node, next_n] - c[prev_n, next_n] + time_penalty
            
            best_idx = np.argmin(deltas)
            route.insert(best_idx, node)
        return route


    def repair_regret_2(route, unassigned):
        while unassigned:
            prev_n = np.array([0] + route)
            next_n = np.array(route + [0])
            U = np.array(unassigned)
            
            times_prev = np.zeros(len(prev_n))
            curr_time = 0.0
            for i in range(1, len(prev_n)):
                p = prev_n[i-1]
                curr = prev_n[i]
                curr_time = max(curr_time, e[p]) + d[p] + c[p, curr]
                times_prev[i] = curr_time
                
            times_prev_exp = times_prev[:, None]
            e_prev = e[prev_n][:, None]
            d_prev = d[prev_n][:, None]
            l_next = l[next_n][:, None]
            
            c_prev_u = c[prev_n[:, None], U[None, :]] 
            c_u_next = c[U[None, :], next_n[:, None]] 
            c_prev_next = c[prev_n, next_n][:, None]
            

            arr_U = np.maximum(times_prev_exp, e_prev) + d_prev + c_prev_u
            viol_U = np.maximum(0, arr_U - l[U][None, :])
            
            start_U = np.maximum(arr_U, e[U][None, :])
            arr_next = start_U + d[U][None, :] + c_u_next
            viol_next = np.maximum(0, arr_next - l_next)
            
            time_penalty = (viol_U + viol_next) * N
            
            deltas = c_prev_u + c_u_next - c_prev_next + time_penalty
            
            if deltas.shape[0] >= 2:
                sorted_2 = np.partition(deltas, 1, axis=0)[:2, :]
                regrets = sorted_2[1, :] - sorted_2[0, :]
            else:
                regrets = np.zeros(len(U))
                
            best_u_idx = np.argmax(regrets)
            best_node = unassigned[best_u_idx]
            best_insert_idx = np.argmin(deltas[:, best_u_idx])
            
            route.insert(best_insert_idx, best_node)
            unassigned.pop(best_u_idx)
            
        return route
    
    def repair_random(route, unassigned):
        for node in unassigned:
            insert_idx = np.random.randint(0, len(route) + 1)
            route.insert(insert_idx, node)
        return route

    destroy_ops = [destroy_random, destroy_segment, destroy_worst]
    repair_ops = [repair_greedy, repair_regret_2, repair_random]
    
    w_d = np.ones(len(destroy_ops))
    w_r = np.ones(len(repair_ops))

    temp_route, temp_cost = greedy(N, e, l, d, c)
    if not temp_route:
        current_route = sorted(range(1, N + 1), key=lambda x: e[x])
    else:
        current_route = temp_route

    current_cost, _ = evaluate_route(current_route)
    best_route, best_cost = current_route[:], current_cost

    T = -(current_cost * 0.1) / math.log(0.6) if current_cost > 0 else 10000.0
    
    cooling_rate = 0.995
    q_size = max(5, int(N * 0.15))
    not_improved = 0
    
    SCORE_GLOBAL_BEST = 33
    SCORE_IMPROVED = 9
    SCORE_ACCEPTED = 13
    DECAY = 0.8

    while not_improved < max_no_improvements:
        p_d = w_d / w_d.sum()
        p_r = w_r / w_r.sum()
        
        d_idx = np.random.choice(len(destroy_ops), p=p_d)
        r_idx = np.random.choice(len(repair_ops), p=p_r)

        temp_route, removed_nodes = destroy_ops[d_idx](current_route[:], q_size)
        new_route = repair_ops[r_idx](temp_route, removed_nodes)
        
        new_cost, is_feasible = evaluate_route(new_route)

        score = 0
        
        if new_cost < best_cost and is_feasible:
            best_cost = new_cost
            best_route = new_route[:]
            current_cost = new_cost
            current_route = new_route[:]
            score = SCORE_GLOBAL_BEST
            not_improved = 0
            
        elif new_cost < current_cost:
            current_cost = new_cost
            current_route = new_route[:]   
            score = SCORE_IMPROVED
            not_improved += 1
            
        else:
            acceptance_prob = math.exp(-(new_cost - current_cost) / T) if T > 0 else 0
            if np.random.rand() < acceptance_prob:            
                current_cost = new_cost
                current_route = new_route[:]
                score = SCORE_ACCEPTED
            not_improved += 1

        w_d[d_idx] = w_d[d_idx] * DECAY + score * (1 - DECAY)
        w_r[r_idx] = w_r[r_idx] * DECAY + score * (1 - DECAY)
        
        T *= cooling_rate

    return best_route, best_cost


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

def main():
    inp = sys.stdin.read().strip().splitlines()
    if not inp:
        return

    N = int(inp[0])
    e, l, d = [0], [999999999], [0]
    c = []

    for i in range(N):
        e_i, l_i, d_i = inp[i + 1].split()
        e.append(int(e_i))
        l.append(int(l_i))
        d.append(int(d_i))

    for i in range(N + 1):
        c.append([int(x) for x in inp[i + N + 1].split()])

    l[0] = max(l[i] + d[i] + c[i][0] for i in range(1, N + 1))

    route, cost = ALNS(N, e, l, d, c, max_no_improvements=100)
    
    print(N)
    if route is not None:
        print(*(route))

if __name__ == "__main__":
    main()