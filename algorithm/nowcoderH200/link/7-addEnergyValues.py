# -*- coding: utf-8 -*-
# @Time    : 2023/10/12

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def addEnergyValues(self, l1: ListNode, l2: ListNode) -> ListNode:
        if not l1 and not l2:
            return l1
        if not l1:
            return l2
        if not l2:
            return l1
    
        dummpy = ListNode(0)
        pre = dummpy
        post = 0
        while l1 or l2:
            val1 = l1.val if l1 else 0
            val2 = l2.val if l2 else 0
        
            remainder = (val1 + val2 + post) % 10
            post = (val1 + val2 + post) // 10
            pre.next = ListNode(remainder)
            pre = pre.next
            if l1:
                l1 = l1.next
            if l2:
                l2 = l2.next
    
        pre.next = ListNode(1) if post > 0 else None
    
        return dummpy.next
    
if __name__ == "__main__":
    print(9%10)
    print(9//10)
