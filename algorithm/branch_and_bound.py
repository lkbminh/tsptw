import sys

def BranchAndBound(N, e, l, d, c):
    min_edge = float('inf')
    for i in range(N + 1):
        for j in range(N + 1):
            if i != j:
                min_edge = min(min_edge, c[i][j])

    best_cost = float('inf')
    best_route = []
    visited = [False] * (N + 1)
    route = []

    def backtrack(level, curr, time, cost):
        nonlocal best_cost, best_route

        remaining = N - level
        if cost + (remaining * min_edge) >= best_cost:
            return

        if level == N:
            total_cost = cost + c[curr][0]
            if total_cost < best_cost:
                best_cost = total_cost
                best_route = route[:]
            return

        # Branching
        for nxt in range(1, N + 1):
            if not visited[nxt]:
                arrival = time + c[curr][nxt]
                
                if arrival <= l[nxt]:
                    start = max(arrival, e[nxt])
                    finish = start + d[nxt]
                    
                    visited[nxt] = True
                    route.append(nxt)
                    
                    backtrack(level + 1, nxt, finish, cost + c[curr][nxt])
                    
                    route.pop()
                    visited[nxt] = False

    backtrack(0, 0, 0.0, 0.0)
    return best_route, best_cost

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

    route, cost = BranchAndBound(N, e, l, d, c)
    
    if cost < float('inf'):
        print(N)
        print(*(route))        

if __name__ == "__main__":
    main()