# -*- coding: utf-8 -*-
# @Time    : 2023/9/4

class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        if not head:
            return head
        
        dummpy = ListNode(0)
        dummpy.next = head
        slow = dummpy
        fast = head
        
        while n:
            n -= 1
            fast = fast.next
        
        while fast:
            fast = fast.next
            slow = slow.next
        slow.next = slow.next.next
        
        return dummpy.next


if __name__ == "__main__":
    print()
