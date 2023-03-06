import queue

def bfs(adj, start):
    if len(adj) <= 1:
        return adj

    visited = set()
    q = queue.Queue()
    q.put(start)
    visited.add(start)
    while not q.empty():
        u = q.get()
        print(u)

        for v in adj.get(u, []):
            if v not in visited:
                visited.add(v)
                q.put(v)

    return visited


graph = {1: [4, 2], 2: [3, 4], 3: [4], 4: [5]}
res = bfs(graph, 1)
print(res)
