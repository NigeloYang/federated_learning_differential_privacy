# -*- coding: utf-8 -*-
# @Time    : 2023/10/12
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
        
class Solution:
    def reverseBetween(self , head: ListNode, left: int, right: int) -> ListNode:
        if not head:
            return head
        dummpy = ListNode(0)
        dummpy.next = head
        pre = dummpy
        for i in range(left-1):
            pre = pre.next
        cur = pre.next
        slow = None
        for i in range(right-left+1):
            temp = cur.next
            cur.next = slow
            slow = cur
            cur = temp
        pre.next.next = cur
        pre.next = slow
        return dummpy.next
    
if __name__ == "__main__":
    print()
