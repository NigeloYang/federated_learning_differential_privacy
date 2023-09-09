# -*- coding: utf-8 -*-
# @Time    : 2023/9/9

class Solution:
    def __init__(self):
        self.stack1 = []
        self.minv = []
    
    def push(self, node):
        self.stack1.append(node)
        
        if not self.minv:
            self.minv.append(node)
        elif node < self.minv[-1]:
            self.minv.append(node)
        else:
            self.minv.append(self.minv[-1])
    
    def pop(self):
        self.stack1.pop()
        self.minv.pop()
    
    def top(self):
        return self.stack1[-1]
    
    def min(self):
        return self.minv[-1]


if __name__ == "__main__":
    print()
