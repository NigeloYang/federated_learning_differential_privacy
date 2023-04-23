#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/2/26 13:31
# @File    : d6.py
# @Author  : Richard Yang
'''剑指 Offer 30. 包含 min 函数的栈
定义栈的数据结构，请在该类型中实现一个能够得到栈的最小元素的 min 函数在该栈中，调用 min、push 及 pop 的时间复杂度都是 O(1)。

 

示例:

MinStack minStack = new MinStack();
minStack.push(-2);
minStack.push(0);
minStack.push(-3);
minStack.min();   --> 返回 -3.
minStack.pop();
minStack.top();      --> 返回 0.
minStack.min();   --> 返回 -2.
 

提示：

各函数的调用总次数不超过 20000 次
'''
import math


class MinStack:
    def __init__(self):
        """
        initialize your data structure here.
        """
        self.stack = []
        self.stack_min = [math.inf]
    
    def push(self, x: int) -> None:
        self.stack.append(x)
        self.stack_min.append(min(x, self.stack_min[-1]))
    
    def pop(self) -> None:
        self.stack.pop()
        self.stack_min.pop()
    
    def top(self) -> int:
        if not self.stack:
            return []
        return self.stack[-1]
    
    def min(self) -> int:
        if not self.stack_min:
            return []
        return self.stack_min[-1]


if __name__ == "__main__":
    # Your MinStack object will be instantiated and called as such:
    obj = MinStack()
    obj.push(-2)
    obj.push(0)
    obj.push(-3)
    print(obj.min())
    obj.pop()
    print(obj.top())
    print(obj.min())
