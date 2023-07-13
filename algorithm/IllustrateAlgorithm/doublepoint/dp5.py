# -*- coding: utf-8 -*-
# @Time    : 2023/6/25

'''剑指 Offer 52. 两个链表的第一个公共节点
输入两个链表，找出它们的第一个公共节点。'''


# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        lena, lenb = 0, 0
        a, b = headA, headB
        if a:
            lena += 1
            a = a.next
        if b:
            lenb += 1
            b = b.next
        if lena > lenb:
            d = lena - lenb
            while d:
                headA = headA.next
                d -= 1
        else:
            d = lenb - lena
            while d:
                headB = headB.next
                d -= 1
        
        while headA != headB:
            headA = headA.next
            headB = headB.next
        
        return headA


    def getIntersectionNode2(self, headA: ListNode, headB: ListNode) -> ListNode:
        A, B = headA, headB
        while A != B:
            A = A.next if A else headB
            B = B.next if B else headA
        return A


if __name__ == "__main__":
    pass
