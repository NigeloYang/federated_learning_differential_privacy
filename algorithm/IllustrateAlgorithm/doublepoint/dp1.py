# -*- coding: utf-8 -*-
# @Time    : 2023/6/25

'''剑指 Offer 18. 删除链表的节点
给定单向链表的头指针和一个要删除的节点的值，定义一个函数删除该节点。

返回删除后的链表的头节点。'''


# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    def deleteNode(self, head: ListNode, val: int) -> ListNode:
        if head.val == val:
            return head.next
        pre, cur = head, head.next
        while cur and cur.val != val:
            pre, cur = cur, cur.next
        if cur:
            pre.next = cur.next
        return head
    
    def deleteNode2(self, head: ListNode, val: int) -> ListNode:
        root = ListNode(-1)
        root.next = head
        s = root
        while s.next:
            if s.next.val == val:
                s.next = s.next.next
                break
            s = s.next
        return root.next


if __name__ == "__main__":
    pass
