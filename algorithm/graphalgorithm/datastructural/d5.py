#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/2/26 13:17
# @File    : d5.py
# @Author  : Richard Yang
'''剑指 Offer 24. 反转链表
定义一个函数，输入一个链表的头节点，反转该链表并输出反转后链表的头节点。

示例:
输入: 1->2->3->4->5->NULL
输出: 5->4->3->2->1->NULL
 

限制：
0 <= 节点个数 <= 5000
'''

import linkedlist as ll


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def reverseList(self, head):
        # 列表方式
        # node_list = list()
        # while head:
        #     node_list.append(head.val)
        #     head = head.next
        # res = ListNode()
        # head = res
        # for val in node_list[::-1]:
        #     node = ListNode(val)
        #     head.next = node
        #     head = head.next
        #
        # return res.next
        
        # 头插指针方式
        cur, pre = head, None
        while cur:
            temp = cur.next
            cur.next = pre
            pre = cur
            cur = temp
        return pre


if __name__ == "__main__":
    data1 = [1, 2, 3, 4, 5]
    head = ll.LinkList()
    head.initList(data1)
    print('before reverseList:')
    head.traveList()
    print('after reverseList:')
    pre  = Solution().reverseList(head.head)
    pre.traveList()