# -*- coding: utf-8 -*-
# @Time    : 2023/8/31

class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


# @param head ListNode类
# @param m int整型
# @param n int整型
# @return ListNode类

class Solution:
    def reverseKGroup(self, head: ListNode, k: int) -> ListNode:
        tail = head
        
        # 判定是否能充分分组
        for _ in range(k):
            if not tail:
                return tail
            tail = tail.next
        
        pre = None
        cur = head
        
        # 反转，调整节点断掉顺序
        while cur != tail:
            temp = cur.next
            
            # 头插法
            cur.next = pre
            pre = cur
            
            # 更新节点
            cur = temp
        
        head.next = self.reverseKGroup(tail,k)
        return pre
if __name__ == "__main__":
    print()
