# -*- coding: utf-8 -*-
# @Time    : 2023/10/13

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        if not lists:
            return None
        return self.merge_sort(lists, 0, len(lists) - 1)
    
    def merge_sort(self, lists: List[ListNode], l: int, r: int) -> ListNode:
        if l == r:
            return lists[l]
        mid = (l + r) // 2
        l = self.merge_sort(lists, l, mid)
        r = self.merge_sort(lists, mid + 1, r)
        return self.merge(l, r)
    
    def merge(self, a: ListNode, b: ListNode) -> ListNode:
        dummpy = ListNode(-1)
        cur = dummpy
        while a and b:
            if a.val < b.val:
                cur.next = a
                a = a.next
            else:
                cur.next = b
                b = b.next
            cur = cur.next
        if a:
            cur.next = a
        if b:
            cur.next = b
        return dummpy.next


if __name__ == "__main__":
    print()
