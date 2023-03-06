'''最小栈
设计一个支持 push ，pop ，top 操作，并能在常数时间内检索到最小元素的栈。
实现 MinStack 类:
MinStack() 初始化堆栈对象。
void push(int val) 将元素val推入堆栈。
void pop() 删除堆栈顶部的元素。
int top() 获取堆栈顶部的元素。
int getMin() 获取堆栈中的最小元素。

示例 1:
输入：
["MinStack","push","push","push","getMin","pop","top","getMin"]
[[],[-2],[0],[-3],[],[],[],[]]

输出：
[null,null,null,null,-3,null,0,-2]

解释：
MinStack minStack = new MinStack();
minStack.push(-2);
minStack.push(0);
minStack.push(-3);
minStack.getMin();   --> 返回 -3.
minStack.pop();
minStack.top();      --> 返回 0.
minStack.getMin();   --> 返回 -2.


提示：
-2^31<= val <= 2^31- 1
pop、top 和 getMin 操作总是在 非空栈 上调用
push,pop,top, andgetMin最多被调用3 * 10^4次
'''

import math


class MinStack(object):
    # def __init__(self):
    #     self.stack = []
    #
    # def push(self, val):
    #     self.stack.append(val)
    #
    # def pop(self):
    #     self.stack.pop()
    #
    # def top(self):
    #     return self.stack[-1]
    #
    # def getMin(self):
    #     return min(self.stack)
    def __init__(self):
        self.stack = []
        self.min_stack = [math.inf]
    
    def push(self, val):
        self.stack.append(val)
        self.min_stack.append(min(val, self.min_stack[-1]))
    
    def pop(self):
        self.stack.pop()
        self.min_stack.pop()
    
    def top(self):
        return self.stack[-1]
    
    def getMin(self):
        return self.min_stack[-1]


if __name__ == '__main__':
    # Your MinStack object will be instantiated and called as such:
    obj = MinStack()
    obj.push(0)
    obj.push(1)
    obj.push(2)
    obj.push(3)
    obj.pop()
    param_3 = obj.top()
    print()
    param_4 = obj.getMin()
