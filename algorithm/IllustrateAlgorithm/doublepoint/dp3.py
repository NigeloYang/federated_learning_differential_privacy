# -*- coding: utf-8 -*-
# @Time    : 2023/6/25

'''剑指 Offer 22. 链表中倒数第 k 个节点
输入一个链表，输出该链表中倒数第k个节点。为了符合大多数人的习惯，本题从1开始计数，即链表的尾节点是倒数第1个节点。

例如，一个链表有 6 个节点，从头节点开始，它们的值依次是 1、2、3、4、5、6。这个链表的倒数第 3 个节点是值为 4 的节点。

ex1
给定一个链表: 1->2->3->4->5, 和 k = 2.
返回链表 4->5.
'''

# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def getKthFromEnd(self, head: ListNode, k: int) -> ListNode:
        if head:
            return head
        pre = head
        count = 0
        while pre:
            count += 1
            pre = pre.next
        s = head
        while s:
            if count == k:
                return s
            s = s.next
            count -= 1

    def getKthFromEnd2(self, head: ListNode, k: int) -> ListNode:
        fast = head
        low = head
        for i in range(k):
            fast = fast.next
        while fast:
            fast = fast.next
            low = low.next
        return low

if __name__ == "__main__":
    pass
