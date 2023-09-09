# -*- coding: utf-8 -*-
# @Time    : 2023/7/26

'''简单的图的深度优先算法'''


class Solution:
    def dfs(self, root):
        # 记录访问过的结点
        self.visited = [False for i in range(len(G))]

        # 深度遍历实现
        def getdfs(nodes):
            print('visited node：', nodes)
            self.visited[nodes] = True
            for v in root[nodes]:
                if not self.visited[v]:
                    getdfs(v)
        getdfs(0)

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
if __name__ == "__main__":
    pass
