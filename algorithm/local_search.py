import sys

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

                    H = c[i][j] + wait + 0.35 * slack  # Example heuristic function
                    candidates.append((H, j, start, finish))
            
        if not candidates:
            return [], float('inf')
        
        h, j, s, f = min(candidates, key= lambda x: x[0])

        time = s
        cost += c[i][j]
        i = j

        route.append(j)
        visited[j] = 1

    cost += c[i][0] #Back to depot

    return route, cost

def feasible(route, e, l, d, c, start_idx, known_time, known_cost, best_cost):
    curr_time = known_time
    curr_cost = known_cost
    
    curr_node = route[start_idx - 1] if start_idx > 0 else 0

    for idx in range(start_idx, len(route)):
        node = route[idx]
        arrival = curr_time + c[curr_node][node]
        
        if arrival > l[node]:
            return False, float('inf')
            
        start = max(arrival, e[node])
        curr_time = start + d[node]
        curr_cost += c[curr_node][node]
        
        if curr_cost >= best_cost:
            return False, float('inf')
            
        curr_node = node
    
    curr_cost += c[curr_node][0]

    if curr_cost < best_cost:
        return True, curr_cost
    return False, float('inf')


def LocalSearch(N, e, l, d, c):
    best_route, best_cost = greedy(N, e, l, d, c)
    
    if not best_route:
        return [], float('inf')
        
    num_nodes = len(best_route)
    improvement_found = True

    while improvement_found:
        improvement_found = False
        
        departure_times = [0] * num_nodes
        accumulated_costs = [0] * num_nodes
        
        t = 0
        cost = 0
        prev = 0

        for idx in range(num_nodes):
            node = best_route[idx]
            arr = t + c[prev][node]
            t = max(arr, e[node]) + d[node]
            cost += c[prev][node]
            departure_times[idx] = t
            accumulated_costs[idx] = cost
            prev = node

        for i in range(0, num_nodes - 1):
            known_time = departure_times[i-1] if i > 0 else 0
            known_cost = accumulated_costs[i-1] if i > 0 else 0
            
            for j in range(i + 1, num_nodes):
                neighbor_route = best_route.copy()
                neighbor_route[i], neighbor_route[j] = neighbor_route[j], neighbor_route[i]
                
                feasibility, new_cost = feasible(
                    neighbor_route, e, l, d, c, 
                    start_idx=i, 
                    known_time=known_time, 
                    known_cost=known_cost, 
                    best_cost=best_cost
                )
                
                if feasibility:
                    best_route = neighbor_route
                    best_cost = new_cost
                    improvement_found = True
                    break 
            
            if improvement_found:
                break

    return best_route, best_cost

def main():
    inp = sys.stdin.read().strip().splitlines()

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

    route,cost = LocalSearch(N, e, l, d, c)

    print(N)
    for i in route:
        print(i, end = ' ')
    print()

if __name__ == "__main__":
    main()