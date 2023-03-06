'''回文链表
给你一个单链表的头节点 head ，请你判断该链表是否为回文链表。如果是，返回 true ；否则，返回 false 。

示例 1：
输入：head = [1,2,2,1]
输出：true

示例 2：
输入：head = [1,2]
输出：false

提示：
链表中节点数目在范围[1, 105] 内
0 <= Node.val <= 9

'''

import linkedlist as ll


# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def isPalindrome(self, head):
        temp = []
        while head:
            temp.append(head.val)
            head = head.next
        length = len(temp)
        # if length == 1:
        #     return False
        for i in range(int(length/2)):
            if temp[i] != temp[length - i-1]:
                return False
        return True

if __name__ == '__main__':
    data1 = [1, 2, 2, 1]
    data2 = [1, 2]
    head = ll.LinkList()
    head2 = ll.LinkList()
    head.initList(data1)
    head2.initList(data2)
    head.traveList()
    head2.traveList()
    
    print(Solution().isPalindrome(head.head))
    print(Solution().isPalindrome(head2.head))
