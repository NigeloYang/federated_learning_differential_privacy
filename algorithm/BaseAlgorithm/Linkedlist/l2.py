'''删除链表的倒数第N个节点
给你一个链表，删除链表的倒数第 n 个结点，并且返回链表的头结点。
示例 1：
输入：head = [1,2,3,4,5], n = 2
输出：[1,2,3,5]
示例 2：
输入：head = [1], n = 1
输出：[]
示例 3：
输入：head = [1,2], n = 1
输出：[1]

提示：
链表中结点的数目为 sz
1 <= sz <= 30
0 <= Node.val <= 100
1 <= n <= sz
'''

import linkedlist as ll

class Solution:
    def removeNthFromEnd(self, head, n):
        '方案1'
        # start = head
        # end = head
        #
        # for i in range(n+1):
        #     end = end.next
        #
        # while end:
        #     end = end.next
        #     start = start.next
        # start.next = start.next.next
        
        '方案2'
        dumpy = ll.ListNode(0, head)
        start = dumpy
        end = head
        
        for i in range(n):
            end = end.next
        
        while end:
            end = end.next
            start = start.next
        start.next = start.next.next
        
        return head


if __name__ == "__main__":
    data1 = [1, 2, 3, 4, 5]
    data2 = [1]
    n = 2
    head = ll.LinkList()
    head.initList(data1)
    # head.initList(data2)
    print('before remove:')
    head.traveList()
    Solution().removeNthFromEnd(head.head, n)
    print('after remove:')
    head.traveList()
