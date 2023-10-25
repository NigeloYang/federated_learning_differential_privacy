# -*- coding: utf-8 -*-
# @Time    : 2023/10/13
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
        
class Solution:
    def sortList(self, head: ListNode) -> ListNode:
        if not head:
            return head
        pre = head
        res = []
        while pre:
            res.append(pre.val)
            pre = pre.next
        res.sort()
        dummpy = ListNode(0)
        cur = dummpy
        for val in res:
            cur.next = ListNode(val)
            cur = cur.next
        
        return dummpy.next

    def sortList2(self, head: ListNode) -> ListNode:
        return self.merge(head) if head else head

    def merge(self, head):
        if not head or not head.next:
            return head
        fast, slow = head, head
        while fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next
        temp = slow.next
        slow.next = None
        left = self.merge(head)
        right = self.merge(temp)
        return self.merge_sort(left, right)

    def merge_sort(self, a, b):
        dummpy = ListNode(-1)
        cur = dummpy
        while a and b:
            if a.val < b.val:
                cur.next = a
                a = a.next
            else:
                cur.next = b
                b = b.next
            cur = cur.next
        if a:
            cur.next = a
        if b:
            cur.next = b
        return dummpy.next
if __name__ == "__main__":
    print()
