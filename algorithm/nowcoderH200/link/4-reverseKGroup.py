# -*- coding: utf-8 -*-
# @Time    : 2023/10/5

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def reverseKGroup(self, head: ListNode, k: int) -> ListNode:
        if not head:
            return head
        dummpy = ListNode(0)
        dummpy.next = head
        slow = dummpy
        fast = dummpy.next
        count = 0
        while fast:
            count += 1
            if count % k == 0:
                slow = self.reverseNode(slow, fast.next)
                fast = slow.next
            else:
                fast = fast.next
        return dummpy.next
    
    def reverseNode(self, slow, end):
        cur = slow.next
        tail = cur
        while cur != end:
            temp = cur.next
            cur.next = slow.next
            slow.next = cur
            cur = temp
        tail.next = end
        return tail

    def reverseKGroup2(self, head: ListNode, k: int) -> ListNode:
        if not head:
            return head
        count = 0
        cur = head
        while cur:
            cur = cur.next
            count += 1
        fast = dummpy = ListNode(next=head)
        while count >= k:
            count -= k
            cur = fast.next
            slow = None
            for i in range(k):
                temp = cur.next
                cur.next = slow
                slow = cur
                cur = temp
            temp = fast.next
            fast.next.next = cur
            fast.next = slow
            fast = temp
        return dummpy.next

if __name__ == "__main__":
    print()
