# -*- coding: utf-8 -*-
# @Time    : 2023/8/31

class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

# @param pHead1 ListNode类
# @param pHead2 ListNode类
# @return ListNode类

class Solution:
    
    def Merge(self, pHead1: ListNode, pHead2: ListNode) -> ListNode:
        if not pHead1:
            return pHead2
        if not pHead2:
            return pHead1
        
        dummpy = ListNode(0)
        
        pre = dummpy
        
        while pHead1 and pHead2:
            if pHead1.val < pHead2.val:
                pre.next = pHead1
                pHead1 = pHead1.next
            else:
                pre.next = pHead2
                pHead2 = pHead2.next
            pre = pre.next
        
        pre.next = pHead1 if pHead1 else pHead2
        
        return dummpy.next


if __name__ == "__main__":
    print()
