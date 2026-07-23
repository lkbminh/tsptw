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

    route,cost = greedy(N, e, l, d, c)

    print(N)
    for i in route:
        print(i, end = ' ')
    print()

if __name__ == "__main__":
    main()