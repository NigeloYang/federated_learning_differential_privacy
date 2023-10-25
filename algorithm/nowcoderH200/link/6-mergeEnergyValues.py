# -*- coding: utf-8 -*-
# @Time    : 2023/10/12

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def mergeEnergyValues(self, l1: ListNode, l2: ListNode) -> ListNode:
        if not l1 and not l2:
            return l1
        if not l1:
            return l2
        if not l2:
            return l1
        
        dummpy = ListNode(0)
        pre = dummpy
        while l1 and l2:
            if l1.val > l2.val:
                pre.next = l1
                pre = pre.next
                l1 = l1.next
            else:
                pre.next = l2
                pre = pre.next
                l2 = l2.next
        
        if l1:
            pre.next = l1
        else:
            pre.next = l2
        
        return dummpy.next
    
if __name__ == "__main__":
    print()
