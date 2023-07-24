# -*- coding: utf-8 -*-
# @Time    : 2023/7/13

'''排序链表
给你链表的头结点 head ，请将其按 升序 排列并返回 排序后的链表 。'''


# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def sortList(self, head):
        if not head:
            return head
        
        res = []
        cur = head
        while cur:
            res.append(cur.val)
            cur = cur.next
        res.sort()
        cur = head
        for rs in res:
            cur.val = rs
            cur = cur.next
        return head
    
    def sortList2(self, head: ListNode) -> ListNode:
        # 归并排序： 从低至上直接合并排序
        h, length, intv = head, 0, 1
        while h:
            h, length = h.next, length + 1
        res = ListNode(0)
        res.next = head
        # merge the list in different intv.
        while intv < length:
            pre, h = res, res.next
            while h:
                # get the two merge head `h1`, `h2`
                h1, i = h, intv
                while i and h:
                    h, i = h.next, i - 1
                if i:
                    break  # no need to merge because the `h2` is None.
                h2, i = h, intv
                while i and h:
                    h, i = h.next, i - 1
                c1, c2 = intv, intv - i  # the `c2`: length of `h2` can be small than the `intv`.
                # merge the `h1` and `h2`.
                while c1 and c2:
                    if h1.val < h2.val:
                        pre.next, h1, c1 = h1, h1.next, c1 - 1
                    else:
                        pre.next, h2, c2 = h2, h2.next, c2 - 1
                    pre = pre.next
                pre.next = h1 if c1 else h2
                while c1 > 0 or c2 > 0:
                    pre, c1, c2 = pre.next, c1 - 1, c2 - 1
                pre.next = h
            intv *= 2
        return res.next
    
    def sortList3(self, head: ListNode) -> ListNode:
        # 归并排序：递归分割在合并
        if not head and not head.next:
            return head
        
        # 寻找中间节点
        slow, fast = head, head.next
        while fast and fast.next:
            fast, slow = fast.next.next, slow.next
        
        mid, slow.next = slow, None
        
        # 递归分割区间
        left, right = self.sortList3(head), self.sortList3(mid)
        
        # 合并排序
        h = res = ListNode(0)
        while left and right:
            if left.val < right.val:
                h.next, left = left, left.next
            else:
                h.next, right = right, right.next
            h = h.next
        h.next = left if left else right
        return res.next


if __name__ == "__main__":
    pass
