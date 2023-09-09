# -*- coding: utf-8 -*-
# @Time    : 2023/9/5

class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
        
class Solution:
    def isPail(self, head: ListNode) -> bool:
        if not head:
            return True
        
        cur = head
        vals = []
        while cur:
            vals.append(cur.val)
            cur = cur.next
        
        mid = int(len(vals) / 2)
        
        if len(vals) % 2 == 1:
            for i in range(mid):
                if vals[mid - i - 1] != vals[mid + i + 1]:
                    return False
        else:
            for i in range(mid):
                if vals[mid - i - 1] != vals[mid + i]:
                    return False
        
        return True

if __name__ == "__main__":
    print()
