# -*- coding: utf-8 -*-
# @Time    : 2023/10/12

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
        
class Solution:
    def hasCycle(self , head: ListNode) -> bool:
        if not head or not head.next:
            return False
        slow = head
        fast = head.next
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow.val == fast.val:
                return True
        return False
    
if __name__ == "__main__":
    print()
