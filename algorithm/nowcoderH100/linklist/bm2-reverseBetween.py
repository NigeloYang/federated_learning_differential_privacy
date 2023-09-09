# -*- coding: utf-8 -*-
# @Time    : 2023/8/30

class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
        
# @param head ListNode类
# @param m int整型
# @param n int整型
# @return ListNode类
#
class Solution:
    def reverseBetween(self, head: ListNode, m: int, n: int) -> ListNode:
        res = ListNode(0)
        res.next = head
        
        pre = res
        cur = head
        
        for _ in range(1, m):
            pre = cur
            cur = cur.next
        
        for _ in range(m, n):
            temp = cur.next
            cur.next = temp.next
            temp.next = pre.next
            pre.next = temp
        
        return res.next

if __name__ == "__main__":
    print()
