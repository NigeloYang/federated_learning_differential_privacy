# -*- coding: utf-8 -*-
# @Time    : 2023/10/2

class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def deleteNodes(self, head: ListNode) -> ListNode:
        if not head:
            return head
        
        pre = head
        cur = head.next
        while cur and cur.next:
            if cur.val > cur.next.val and pre.val < cur.val:
                temp = cur.next
                pre.next = cur.next
                cur = temp
            cur = cur.next
            pre = pre.next
        return head
    
if __name__ == "__main__":
    print()
