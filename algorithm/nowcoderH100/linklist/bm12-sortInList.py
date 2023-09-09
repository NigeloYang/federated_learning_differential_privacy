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
# @param head ListNode类 the head node
# @return ListNode类
#
class Solution:
    def sortInList(self, head: ListNode) -> ListNode:
        if not head:
            return head
        
        pre = head
        vals = []
        while pre:
            vals.append(pre.val)
            pre = pre.next
        
        vals.sort()
        
        dummpy = ListNode(0)
        cur = dummpy
        for val in vals:
            cur.next = ListNode(val)
            cur = cur.next
        
        return dummpy.next
    
    def sortInlist2(self, head):
        if not head or not head.next.next:
            return head
        
        left = head
        mid = head.next
        right = head.next.next
        while right and right.next:
            left = left.next
            mid = mid.next
            right = right.next.next
        
        left.next = None
        
        return self.mergelist(self.sortInList(head), self.sortInList(mid))
    
    def mergelist(self, head1, head2):
        if not head1:
            return head2
        if not head2:
            return head1
        
        dummpy = ListNode(0)
        cur = dummpy
        
        while head1 and head2:
            if head1.val < head2.val:
                cur.next = head1
                head1 = head1.next
            else:
                cur.next = head2
                head2 = head2.next
            cur = cur.next
        
        if head1:
            cur.next = head1
        if head2:
            cur.next = head2
        
        return dummpy.next


if __name__ == "__main__":
    print()
