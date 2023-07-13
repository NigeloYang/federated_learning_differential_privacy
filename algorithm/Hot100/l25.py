# -*- coding: utf-8 -*-
# @Time    : 2023/7/9

''' K 个一组翻转链表
给你链表的头节点 head ，每 k 个节点一组进行翻转，请你返回修改后的链表。

k 是一个正整数，它的值小于或等于链表的长度。如果节点总数不是 k 的整数倍，那么请将最后剩余的节点保持原有顺序。

你不能只是单纯的改变节点内部的值，而是需要实际进行节点交换。'''


# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def reverseKGroup(self, head, k: int):
        if not head:
            return head
        dummpy = ListNode(0)
        dummpy.next = head
        prev = dummpy
        
        while True:
            tail = prev
            for i in range(k):
                tail = tail.next
                if not tail:
                    return dummpy.next
            # 确定下一轮开始的头节点
            temp = tail.next
            # 排序结果
            head, tail = self.revers(head, tail)
            # 嵌入排序后的链表
            prev.next = head
            tail.next = temp
            # 设置下一轮开始的顺序
            prev = tail
            head = temp
        
        return dummpy.next
    
    def revers(self, head, tail):
        pos = tail.next
        p = head
        while pos != tail:
            temp = p.next
            p.next = pos
            pos = p
            p = temp
        return tail, head
    
    def reverseKGroup(self, head, k: int):
        if not head:
            return head
        res = []
        temp = head
        while temp:
            res.append(temp.val)
            temp = temp.next
        
        if len(res) < k:
            return head
        else:
            dummpy = ListNode(0)
            node = dummpy
            for i in range(0, len(res), k):
                curr = []
                for j in range(k):
                    if head:
                        curr.append(head.val)
                        head = head.next
                    else:
                        break
                
                m = curr[::-1] if len(curr) == k else curr
                for val in m:
                    newnode = ListNode(val)
                    node.next = newnode
                    node = node.next
            return dummpy.next


if __name__ == "__main__":
    pass
