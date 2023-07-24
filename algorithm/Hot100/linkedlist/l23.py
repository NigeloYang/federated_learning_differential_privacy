# -*- coding: utf-8 -*-
# @Time    : 2023/7/19

'''23 合并 K 个升序链表
给你一个链表数组，每个链表都已经按升序排列。

请你将所有链表合并到一个升序链表中，返回合并后的链表。'''


# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def mergeKLists(self, lists):
        temps = []
        for list in lists:
            while list:
                temps.append(list.val)
                list = list.next
        head = ListNode(0)
        node = head
        if temps:
            temps.sort()
        else:
            return head.next
        for val in temps:
            tnode = ListNode(val)
            node.next = tnode
            node = node.next
        return head.next


if __name__ == "__main__":
    pass
