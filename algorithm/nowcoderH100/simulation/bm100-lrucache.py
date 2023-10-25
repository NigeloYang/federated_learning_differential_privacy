# -*- coding: utf-8 -*-
# @Time    : 2023/9/30

class ListNode:
    def __init__(self, key=None, value=None):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None


class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.dic = {}
        # 创建两个节点: head and tail
        self.head = ListNode()
        self.tail = ListNode()
        # 初始化链表 head <-> tail
        self.head.next = self.tail
        self.tail.prev = self.head
    
    # 因为get与put操作都可能需要将双向链表中的某个节点移到头部(变成最新访问的)，所以定义一个方法
    def moveNodeToHead(self, key):
        # 找到指定节点并将节点摘出来
        node = self.dic[key]
        node.prev.next = node.next
        node.next.prev = node.prev
        # 将节点移动到头部节点
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node
    
    def addNodeToHeader(self, key, value):
        # 生成一个新节点并将其放到头部节点上
        newnode = ListNode(key, value)
        self.dic[key] = newnode
        newnode.prev = self.head
        newnode.next = self.head.next
        self.head.next.prev = newnode
        self.head.next = newnode
    
    def popTailNode(self):
        lastNode = self.tail.prev
        # 掉链表尾部节点在哈希表的对应项
        self.dic.pop(lastNode.key)
        # 去掉最久没有被访问过的节点，即尾部Tail之前的一个节点
        lastNode.prev.next = self.tail
        self.tail.prev = lastNode.prev
        return lastNode
    
    def get(self, key: int) -> int:
        if key in self.dic:
            # 如果已经在链表中了就把它移到头部（变成最新访问的）
            self.moveNodeToHead(key)
            res = self.dic.get(key)
            return res.value
        else:
            return -1
    
    def put(self, key: int, value: int) -> None:
        if key in self.dic:
            # 如果 key 已经在哈希表中了就不需要在链表中加入新的节点,只需要更新字典该值对应节点的value并将其移动到头部
            self.dic[key].value = value
            self.moveNodeToHead(key)
        else:
            if len(self.dic) >= self.capacity:
                # 若cache容量已满，删除cache中最不常用的节点
                self.popTailNode()
            self.addNodeToHeader(key, value)
if __name__ == "__main__":
    print()
