'''反转链表
给你单链表的头节点 head ，请你反转链表，并返回反转后的链表。


'''

import linkedlist as ll


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def reverseList(self, head):
        node_list = list()
        while head:
            node_list.append(head.val)
            head = head.next
        res = ListNode()
        head = res
        for val in node_list[::-1]:
            node = ListNode(val)
            head.next = node
            head = head.next
        
        return res.next


if __name__ == "__main__":
    data1 = [1, 2, 3, 4, 5]
    head = ll.LinkList()
    head.initList(data1)
    print('before reverseList:')
    head.traveList()
    print('after reverseList:')
    Solution().reverseList(head.head)
