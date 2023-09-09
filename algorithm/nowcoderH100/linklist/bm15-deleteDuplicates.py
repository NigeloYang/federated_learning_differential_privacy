# -*- coding: utf-8 -*-
# @Time    : 2023/9/5

class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
#
# 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
#
#
# @param head ListNode类
# @return ListNode类
#
class Solution:
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        if not head:
            return head
        
        res = set()
        cur = head
        
        dummpy = ListNode(0)
        pre = dummpy
        pre.next = head
        
        while cur:
            if cur.val not in res:
                res.add(cur.val)
                pre = pre.next
                cur = cur.next
            else:
                temp = cur.next
                pre.next = cur.next
                cur = temp
        return head
    
if __name__ == "__main__":
    print()
