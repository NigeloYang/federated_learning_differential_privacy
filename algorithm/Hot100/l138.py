# -*- coding: utf-8 -*-
# @Time    : 2023/7/13

'''复制带随机指针的链表'''


# Definition for a Node.

class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random


class Solution:
    def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
        Mydic = dict()
        
        def recursion(node: 'Node') -> 'Node':
            if node is None: return None
            if node in Mydic: return Mydic.get(node)
            root = Node(node.val)
            Mydic[node] = root
            root.next = recursion(node.next)
            root.random = recursion(node.random)
            return root
        
        return recursion(head)
    
    def copyRandomList2(self, head: 'Optional[Node]') -> 'Optional[Node]':
        if not head:
            return head
        resdic = dict()
        
        temp1 = head
        while temp1:
            cur = Node(temp1.val)
            resdic[temp1] = cur
            temp1 = temp1.next
        temp2 = head
        while temp2:
            resdic.get(temp2).next = resdic.get(temp2.next)
            resdic.get(temp2).random = resdic.get(temp2.random)
            temp2 = temp2.next
        return resdic.get(head)
        
        
if __name__ == "__main__":
    pass
