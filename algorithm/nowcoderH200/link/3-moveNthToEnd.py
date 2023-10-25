# -*- coding: utf-8 -*-
# @Time    : 2023/10/2
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
        
class Solution:
    def moveNthToEnd(self, head: ListNode, n: int) -> ListNode:
        if not head:
            return head
        if n == 1:
            return head
        dummpy = ListNode(0)
        dummpy.next = head
        fast = dummpy
        slow = dummpy
        while n > 0:
            fast = fast.next
            n -= 1
        while fast.next:
            fast = fast.next
            slow = slow.next
        
        cur = slow.next
        slow.next = slow.next.next
        cur.next = None
        fast.next = cur
        return dummpy.next

if __name__ == "__main__":
    print()
