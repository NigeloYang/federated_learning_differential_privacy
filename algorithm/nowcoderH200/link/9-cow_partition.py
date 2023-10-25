# -*- coding: utf-8 -*-
# @Time    : 2023/10/13

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def cow_partition(self, head: ListNode, x: int) -> ListNode:
        if not head:
            return head
        
        dummpy = ListNode(0)
        left = dummpy
        
        right = ListNode(0)
        rights = right
        
        while head:
            if head.val < x:
                left.next = head
                left = left.next
            else:
                right.next = head
                right = right.next
            head = head.next
        
        left.next = rights.next
        right.next = None
        return dummpy.next
    
if __name__ == "__main__":
    print()
