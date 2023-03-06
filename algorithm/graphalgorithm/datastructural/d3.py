#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/2/25 13:04
# @File    : d3.py
# @Author  : Richard Yang
'''剑指 Offer 09. 用两个栈实现队列
用两个栈实现一个队列。队列的声明如下，请实现它的两个函数 appendTail 和 deleteHead ，
分别完成在队列尾部插入整数和在队列头部删除整数的功能。(若队列中没有元素，deleteHead 操作返回 -1 )

示例 1：
输入：
["CQueue","appendTail","deleteHead","deleteHead","deleteHead"]
[[],[3],[],[],[]]
输出：[null,null,3,-1,-1]

示例 2：
输入：
["CQueue","deleteHead","appendTail","appendTail","deleteHead","deleteHead"]
[[],[],[5],[2],[],[]]
输出：[null,-1,null,null,5,2]

提示：
1 <= values <= 10000
最多会对 appendTail、deleteHead 进行 10000 次调用
'''


class CQueue:
    def __init__(self):
        self.stack_in = []
        self.stack_out = []
    
    def appendTail(self, value: int) -> None:
        self.stack_in.append(value)
    
    def deleteHead(self) -> int:
        if not self.stack_out:
            if not self.stack_in:
                return -1
            else:
                while self.stack_in:
                    self.stack_out.append(self.stack_in.pop())
        return self.stack_out.pop()


if __name__ == "__main__":
    # Your CQueue object will be instantiated and called as such:
    obj = CQueue()
    obj.appendTail(1)
    param_2 = obj.deleteHead()
    print(param_2)
