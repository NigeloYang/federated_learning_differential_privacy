'''删除链表中的节点
有一个单链表的head，我们想删除它其中的一个节点node。

给你一个需要删除的节点node。你将无法访问第一个节点head。

链表的所有值都是 唯一的，并且保证给定的节点node不是链表中的最后一个节点。

删除给定的节点。注意，删除节点并不是指从内存中删除它。这里的意思是：

给定节点的值不应该存在于链表中。
链表中的节点数应该减少 1。
node前面的所有值顺序相同。
node后面的所有值顺序相同。
自定义测试：
对于输入，你应该提供整个链表head和要给出的节点node。node不应该是链表的最后一个节点，而应该是链表中的一个实际节点。
我们将构建链表，并将节点传递给你的函数。
输出将是调用你函数后的整个链表。

输入：head = [4,5,1,9], node = 5
输出：[4,1,9]
解释：指定链表中值为5的第二个节点，那么在调用了你的函数之后，该链表应变为 4 -> 1 -> 9

输入：head = [4,5,1,9], node = 1
输出：[4,5,9]
解释：指定链表中值为1的第三个节点，那么在调用了你的函数之后，该链表应变为 4 -> 5 -> 9
'''
import linkedlist as ll


# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    def deleteNode(self, node):
        """
        :type node: ListNode
        :rtype: void Do not return anything, modify node in-place instead.
        """
        node.val = node.next.val
        node.next = node.next.next


if __name__ == "__main__":
    data1 = [4, 5, 1, 9]
    node = 5
    head = ll.LinkList()
    head.initList(data1)
    print('before deleteNode:')
    head.traveList()
    
    temp = head.head
    while temp:
        if temp.val == node:
            break
        else:
            temp = temp.next
    print('current node: ', temp.val)
    Solution().deleteNode(temp)
