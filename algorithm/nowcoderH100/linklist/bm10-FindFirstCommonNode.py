# -*- coding: utf-8 -*-
# @Time    : 2023/9/4

class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
#
#
#
# @param pHead1 ListNode类
# @param pHead2 ListNode类
# @return ListNode类
#
class Solution:
    def FindFirstCommonNode(self, pHead1, pHead2):
        if not pHead1 or not pHead2:
            return
        
        cur1 = pHead1
        cur2 = pHead2
        
        while cur1 != cur2:
            cur1 = cur1.next if cur1 else pHead2
            cur2 = cur2.next if cur2 else pHead1
        
        return cur1

if __name__ == "__main__":
    print()
