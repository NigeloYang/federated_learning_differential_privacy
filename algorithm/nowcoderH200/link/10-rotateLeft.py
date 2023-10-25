# -*- coding: utf-8 -*-
# @Time    : 2023/10/13

class Solution:
    def rotateLeft(self, head: ListNode, k: int) -> ListNode:
        if not head:
            return head
        count = 0
        cur = head
        while cur:
            count += 1
            cur = cur.next
        k %= count
        
        fast = head
        for i in range(k):
            fast = fast.next
        slow = head
        while fast.next:
            fast = fast.next
            slow = slow.next
        
        fast.next = head
        rightl = slow.next
        slow.next = None
        
        return rightl
    
if __name__ == "__main__":
    print()
