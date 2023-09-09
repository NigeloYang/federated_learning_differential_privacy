# -*- coding: utf-8 -*-
# @Time    : 2023/8/31
from typing import List


class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        length = len(lists)

        if length == 0:
            return None
        if length == 1:
            return lists[0]

        mid = length // 2

        return self.Merge(self.mergeKLists(lists[:mid]), self.mergeKLists(lists[mid:]))

    def Merge(self, pHead1, pHead2):
        if not pHead1:
            return pHead2
        if not pHead2:
            return pHead1
    
        dummpy = ListNode(0)
    
        cur = dummpy
    
        while pHead1 and pHead2:
            if pHead1.val < pHead2.val:
                cur.next = pHead1
                pHead1 = pHead1.next
            else:
                cur.next = pHead2
                pHead2 = pHead2.next
            cur = cur.next
    
        cur.next = pHead1 if pHead1 else pHead2
    
        return dummpy.next


if __name__ == "__main__":
    print()
