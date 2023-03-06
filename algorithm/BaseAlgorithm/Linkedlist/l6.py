'''环形链表
给你一个链表的头节点 head ，判断链表中是否有环。
如果链表中有某个节点，可以通过连续跟踪 next 指针再次到达，则链表中存在环。 为了表示给定链表中的环，评测系统内部使用整数 pos
来表示链表尾连接到链表中的位置（索引从 0 开始）。注意：pos 不作为参数进行传递。仅仅是为了标识链表的实际情况。
如果链表中存在环，则返回 true 。 否则，返回 false 。

示例 1：
输入：head = [3,2,0,-4], pos = 1
输出：true
解释：链表中有一个环，其尾部连接到第二个节点。

示例2：
输入：head = [1,2], pos = 0
输出：true
解释：链表中有一个环，其尾部连接到第一个节点。

示例 3：
输入：head = [1], pos = -1
输出：false
解释：链表中没有环。

提示：
链表中节点的数目范围是 [0, 104]
-105 <= Node.val <= 105
pos 为 -1 或者链表中的一个 有效索引 。
'''

import linkedlist as ll


# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    def hasCycle(self, head):
        # 第一种，如果链表有重复数字出现则失效
        # seta = set()
        # p = head
        # while p:
        #     seta.add(p.val)
        #     p = p.next
        #     if p.val in seta:
        #         return True
        # return False
        
        # 快慢指针的相遇的
        fast = head
        slow = head
        while (fast and fast.next):
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                return True
        return False


if __name__ == "__main__":
    data1 = [3, 2, 0, -4]
    pos1 = 1
    data2 = [1, 2]
    pos2 = 0
    data3 = [1]
    pos3 = -1
    
    head = ll.LinkList()
    head.initList_c(data1)
    head.traveList_c()
    arr = [-21, 10, 17, 8, 4, 26, 5, 35, 33, -7, -16, 27, -12, 6, 29, -12, 5, 9, 20, 14, 14, 2, 13, -24, 21, 23, -21, 5]
    arr_set = set(arr)
    print(len(arr_set))
    print(len(arr))
