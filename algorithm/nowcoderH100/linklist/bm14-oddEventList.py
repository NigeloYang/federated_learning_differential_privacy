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
    def oddEvenList(self, head: ListNode) -> ListNode:
        if not head:
            return head
        
        cur = head
        
        dummpy1 = ListNode(0)
        pre1 = dummpy1
        
        dummpy2 = ListNode(0)
        cur2, pre2 = dummpy2, dummpy2
        count = 0
        while cur:
            count += 1
            if count % 2 == 1:
                pre1.next = cur
                pre1 = pre1.next
            else:
                pre2.next = cur
                pre2 = pre2.next
            cur = cur.next
        
        pre2.next = None
        pre1.next = cur2.next
        
        return dummpy1.next

if __name__ == "__main__":
    print()
