# -*- coding: utf-8 -*-
# @Time    : 2023/9/4

class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

# 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
#
#
# @param head1 ListNode类
# @param head2 ListNode类
# @return ListNode类
#
class Solution:
    def reverslist(self, phead: ListNode) -> ListNode:
        cur = phead
        pre = None
        while cur:
            temp = cur.next
            cur.next = pre
            pre = cur
            cur = temp
        
        return pre
    
    def addInList(self, head1: ListNode, head2: ListNode) -> ListNode:
        if not head1:
            return head2
        if not head2:
            return head1
        
        c1 = self.reverslist(head1)
        c2 = self.reverslist(head2)
        res = ListNode(-1)
        pre = res
        
        temp = 0
        while c1 or c2 or temp != 0:
            val1 = c1.val if c1 else 0
            val2 = c2.val if c2 else 0
            val = val1 + val2 + temp
            temp = int(val / 10)
            val = val % 10
            
            pre.next = ListNode(val)
            pre = pre.next
            if c1:
                c1 = c1.next
            if c2:
                c2 = c2.next
        
        return self.reverslist(res.next)



if __name__ == "__main__":
    print()
