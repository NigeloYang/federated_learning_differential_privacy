# -*- coding: utf-8 -*-
# @Time    : 2023/7/7

'''相交链表
给你两个单链表的头节点 headA 和 headB ，请你找出并返回两个单链表相交的起始节点。如果两个链表不存在相交节点，返回 null 。

图示两个链表在节点 c1 开始相交：'''


# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode):
        if not headA or not headB:
            return headA
        a, b = headA, headA
        lena, lenb = 0, 0
        
        while a:
            a = a.next
            lena += 1
        while b:
            b = b.next
            lenb += 1
        
        diff = lena - lenb
        A, B = headA, headA
        while A != B:
            if diff > 0:
                A = A.next
                diff -= 1
            elif diff < 0:
                B = B.next
                diff += 1
            else:
                A = A.next
                B = B.next
        return A
    
    def getIntersectionNode2(self, headA: ListNode, headB: ListNode):
        A, B = headA, headB
        while A != B:
            A = A.next if A else headB
            B = B.next if B else headA
        return A


if __name__ == "__main__":
    pass
