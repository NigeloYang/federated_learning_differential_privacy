# -*- coding: utf-8 -*-
# @Time    : 2023/7/26

'''简单的图的广度优先算法'''


class Solution:
    def bfs(self, root):
        # 记录访问过的结点
        visited = [False for i in range(len(G))]
        
        # 创建队列存放访问节点的顺序
        visited_queue = [0]
        visited[0] = True
        
        while visited_queue:
            nodeidx = visited_queue.pop(0)
            print('visited node：', nodeidx)
            for v in root[nodeidx]:
                if not visited[v]:
                    visited_queue.append(v)
                    visited[v] = True


if __name__ == "__main__":
    # G 是临接表的方式表达图形
    G = {}
    G[0] = [1, 2]
    G[1] = [2, 3, 5]
    G[2] = [0, 1, 3]
    G[3] = [2, 4, 5]
    G[4] = [1, 3, 5]
    G[5] = [1, 3, 4]
    print(G)
    print(Solution().bfs(G))
