# -*- coding: utf-8 -*-
# @Time    : 2023/8/15

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
    def ReverseList(self, head: ListNode) -> ListNode:
        if not head:
            return head
        
        res = []
        temp = head
        
        while temp:
            res.append(temp.val)
            temp = temp.next
        
        dummpy = ListNode(0)
        temp = dummpy
        for val in res[::-1]:
            node = ListNode(val)
            temp.next = node
            temp = temp.next
        
        return dummpy.next
    
    def ReverseList2(self, head: ListNode) -> ListNode:
        if head is None:
            return head
        else:
            p0 = None
            p1 = head
            while p1 is not None:
                tmp = p1.next
                p1.next = p0
                p0 = p1
                p1 = tmp
            return p0


if __name__ == "__main__":
    print()
