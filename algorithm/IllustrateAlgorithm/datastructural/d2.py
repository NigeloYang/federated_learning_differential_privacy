#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/2/25 12:52
# @File    : d2.py
# @Author  : Richard Yang
import linkedlist as ll

# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def reversePrint(self, head: ListNode):
        node_list = []
        while head:
            node_list.append(head.val)
            head = head.next
        return node_list[::-1]


if __name__ == "__main__":
    data1 = [1, 2, 3, 4, 5]
    head = ll.LinkList()
    head.initList(data1)
    print('before reverseList:')
    head.traveList()
    print('after reverseList:')
    print(Solution().reversePrint(head.head))
