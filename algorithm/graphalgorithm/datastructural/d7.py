#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/2/26 13:47
# @File    : d7.py
# @Author  : Richard Yang

'''剑指 Offer 35. 复杂链表的复制
请实现 copyRandomList 函数，复制一个复杂链表。在复杂链表中，每个节点除了有一个 next 指针指向下一个节点，还有一个 random 指针指向链表中的任意节点或者 null。

示例 1：
输入：head = [[7,null],[13,0],[11,4],[10,2],[1,0]]
输出：[[7,null],[13,0],[11,4],[10,2],[1,0]]

示例 2：
输入：head = [[1,1],[2,1]]
输出：[[1,1],[2,1]]

示例 3：
输入：head = [[3,null],[3,0],[3,null]]
输出：[[3,null],[3,0],[3,null]]

示例 4：
输入：head = []
输出：[]
解释：给定的链表为空（空指针），因此返回 null。

提示：
-10000 <= Node.val <= 10000
Node.random 为空（null）或指向链表中的节点。
节点数目不超过 1000 。

注意：本题与主站 138 题相同：https://leetcode-cn.com/problems/copy-list-with-random-pointer/
'''


# Definition for a Node.
class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random


class Solution:
    def copyRandomList(self, head: 'Node') -> 'Node':
        if not head:
            return []
        dic = {}
        cur = head
        while cur:
            dic[cur] = Node(cur.val)
            cur = cur.next
        
        cur = head
        while cur:
            dic[cur].next = dic.get(cur.next)
            dic[cur].random = dic.get(cur.random)
            cur = cur.next
        return dic[head]

        # if not head: return
        # cur = head
        # # 1. 复制各节点，并构建拼接链表
        # while cur:
        #     tmp = Node(cur.val)
        #     tmp.next = cur.next
        #     cur.next = tmp
        #     cur = tmp.next
        # # 2. 构建各新节点的 random 指向
        # cur = head
        # while cur:
        #     if cur.random:
        #         cur.next.random = cur.random.next
        #     cur = cur.next.next
        # # 3. 拆分两链表
        # cur = res = head.next
        # pre = head
        # while cur.next:
        #     pre.next = pre.next.next
        #     cur.next = cur.next.next
        #     pre = pre.next
        #     cur = cur.next
        # pre.next = None  # 单独处理原链表尾节点
        # return res  # 返回新链表头节点


if __name__ == "__main__":
    pass
