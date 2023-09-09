# -*- coding: utf-8 -*-
# @Time    : 2023/9/5

class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
#
# @param head ListNode类
# @return ListNode类
#
class Solution:
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        if not head:
            return head
        
        res = {}
        cur = head
        
        while cur:
            if cur.val not in res:
                res.setdefault(cur.val, 1)
            else:
                res[cur.val] += 1
            cur = cur.next
        
        dummpy = ListNode(0)
        cur = dummpy
        
        for k, v in res.items():
            if v == 1:
                cur.next = ListNode(k)
                cur = cur.next
        
        return dummpy.next
    
if __name__ == "__main__":
    print()
