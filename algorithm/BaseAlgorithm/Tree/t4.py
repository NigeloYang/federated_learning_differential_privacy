import TreeNode as trees


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def levelOrder(self, root):
        if root is None:
            return []
        
        queue = []
        queue.append(root)
        res = []
        while len(queue) > 0:
            sublist = []
            print(len(queue))
            for i in range(len(queue)):
                root = queue.pop(0)
                sublist.append(root.val)
                if root.left is not None:
                    queue.append(root.left)
                if root.right is not None:
                    queue.append(root.right)
            res.append(sublist)
        return res


if __name__ == "__main__":
    data = [1, 2, 2, 3, 4, 4, 3]
    data2 = [1, 2, 2, None, 3, None, 3]
    tree1 = trees.Tree()
    tree2 = trees.Tree()
    
    tree1.createTree(data)
    tree1.breadth_travel()
    print()
    tree2.createTree(data2)
    tree2.breadth_travel()
    print('\n 验证二叉搜索树结果:', Solution().levelOrder(tree1.root))
    print('\n 验证二叉搜索树结果:', Solution().levelOrder(tree2.root))
