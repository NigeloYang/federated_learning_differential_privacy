# -*- coding: utf-8 -*-
# @Time    : 2023/8/31

class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def FindKthToTail(self , pHead: ListNode, k: int) -> ListNode:
        cur = pHead
        count = 0
        while cur:
            cur = cur.next
            count += 1
    
        if count < k:
            return cur
        else:
            while count != k:
                pHead = pHead.next
                count -= 1
        
            return pHead

if __name__ == "__main__":
    print()
