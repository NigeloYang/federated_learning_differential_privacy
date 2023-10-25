# -*- coding: utf-8 -*-
# @Time    : 2023/10/13

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
        
class Solution:
    def sortCowsIV(self, head: ListNode) -> ListNode:
        if not head:
            return head
        dummpy = ListNode(0)
        c1 = dummpy
        dummpy2 = ListNode(0)
        c2 = dummpy2
        
        while head:
            if head.val == 0:
                c1.next = head
                c1 = c1.next
            else:
                c2.next = head
                c2 = c2.next
            head = head.next
        
        c1.next = dummpy2.next
        c2.next = None
        return dummpy.next

if __name__ == "__main__":
    print()
