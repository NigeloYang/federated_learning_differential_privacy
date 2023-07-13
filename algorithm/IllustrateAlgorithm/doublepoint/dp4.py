# -*- coding: utf-8 -*-
# @Time    : 2023/6/25

'''剑指 Offer 25. 合并两个排序的链表
输入两个递增排序的链表，合并这两个链表并使新链表中的节点仍然是递增排序的。
'''


# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        if not l1:
            return l2
        if not l2:
            return l1
        temp = ListNode(0)
        sta = temp
        while l1 and l2:
            if l1.val <= l2.val:
                sta.next = l1
                l1 = l1.next
            else:
                sta.next = l2
                l2 = l2.next
            sta = sta.next
        if not l1:
            sta.next = l2
        else:
            sta.next = l1
        return temp.next

    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        if not l1:
            return l2
        if not l2:
            return l1
        if l1.val <= l2.val:
            l1.next = self.mergeTwoLists(l1.next,l2)
            return l1
        else:
            l2.next = self.mergeTwoLists(l1,l2.next)
            return l2


if __name__ == "__main__":
    pass
