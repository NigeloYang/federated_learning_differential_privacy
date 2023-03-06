'''合并两个有序链表
将两个升序链表合并为一个新的 升序 链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。

示例 1：
输入：l1 = [1,2,4], l2 = [1,3,4]
输出：[1,1,2,3,4,4]
示例 2：
输入：l1 = [], l2 = []
输出：[]
示例 3：
输入：l1 = [], l2 = [0]
输出：[0]

提示：
两个链表的节点数目范围是 [0, 50]
-100 <= Node.val <= 100
l1 和 l2 均按 非递减顺序 排列
'''
import linkedlist as ll


# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def mergeTwoLists(self, list1, list2):
        dump = ListNode()
        cur = dump
        if list1 == None:
            return list2
        
        if list2 == None:
            return list1
        
        while (list1 and list2):
            if list1.val <= list2.val:
                cur.next = ListNode(list1.val)
                list1 = list1.next
            else:
                cur.next = ListNode(list2.val)
                list2 = list2.next
            cur = cur.next
        cur.next = list1 if list1 else list2
        return dump.next


if __name__ == '__main__':
    l1 = [1, 2, 4]
    l2 = [1, 3, 4]
    list1 = ll.LinkList()
    list1.initList(l1)
    list2 = ll.LinkList()
    list2.initList(l2)
    # list1.traveList()
    # list2.traveList()
    res = Solution().mergeTwoLists(list1.head, list2.head)
    print(res.val)
    print(res.next.val)
    print(res.next.next.val)
    print(res.next.next.next.val)
    print(res.next.next.next.next.val)
    print(res.next.next.next.next.next.val)
    