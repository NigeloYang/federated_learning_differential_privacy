# -*- coding: utf-8 -*-
# @Time    : 2023/10/13

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
        
class Solution:
    def maxPalindrome(self, head: ListNode) -> ListNode:
        if not head:
            return head
        
        pre = head
        res = []
        while pre:
            res.append(pre.val)
            pre = pre.next
        start, end = 0, 0
        for i in range(len(res)):
            lodd, rodd = self.paliExpand(res, i, i)
            leven, reven = self.paliExpand(res, i, i + 1)
            if rodd - lodd > end - start:
                start = lodd
                end = rodd
            if reven - leven > end - start:
                start = leven
                end = reven
        if start == 0 and end == len(res) - 1:
            return None
        else:
            p = head
            for i in range(start):
                p = p.next
            res = p
            while start < end:
                p = p.next
                start += 1
            p.next = None
            return res
    
    def paliExpand(self, nums, l, r):
        n = len(nums)
        while l >= 0 and r < n and nums[l] == nums[r]:
            l -= 1
            r += 1
        return l + 1, r - 1
    
if __name__ == "__main__":
    print()
