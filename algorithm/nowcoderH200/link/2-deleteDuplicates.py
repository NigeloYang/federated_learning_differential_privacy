# -*- coding: utf-8 -*-
# @Time    : 2023/10/2

class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
        
class Solution:
    def deleteDuplicates(self , head: ListNode) -> ListNode:
        if not head:
            return head
        pre = head
        cur = head.next
        while cur:
            if pre.val == cur.val:
                cur = cur.next
            else:
                pre.next = cur
                cur = cur.next
                pre = pre.next
        pre.next = None
        return head
    
if __name__ == "__main__":
    print()
