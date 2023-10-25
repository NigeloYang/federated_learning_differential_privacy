# -*- coding: utf-8 -*-
# @Time    : 2023/10/13

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
        
class Solution:
    def isPalindrome(self, head: ListNode) -> bool:
        if not head:
            return head
        
        res = []
        while head:
            res.append(head.val)
            head = head.next
        
        return res == res[::-1]
    
if __name__ == "__main__":
    print()
