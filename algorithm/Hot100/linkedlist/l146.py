# -*- coding: utf-8 -*-
# @Time    : 2023/7/19

'''LRU 缓存
请你设计并实现一个满足  LRU (最近最少使用) 缓存 约束的数据结构。
实现 LRUCache 类：
LRUCache(int capacity) 以 正整数 作为容量 capacity 初始化 LRU 缓存
int get(int key) 如果关键字 key 存在于缓存中，则返回关键字的值，否则返回 -1 。
void put(int key, int value) 如果关键字 key 已经存在，则变更其数据值 value ；如果不存在，则向缓存中插入该组 key-value 。如果插入操作导致关键字数量超过 capacity ，则应该 逐出 最久未使用的关键字。
函数 get 和 put 必须以 O(1) 的平均时间复杂度运行。'''


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
            # 如果 key 已经在哈希表中了就不需要在链表中加入新的节点,只需要更新字典该值对应节点的value 并将其移动到头部
            self.dic[key].value = value
            self.moveNodeToHead(key)
        else:
            if len(self.dic) >= self.capacity:
                # 若cache容量已满，删除cache中最不常用的节点
                self.popTailNode()
            self.addNodeToHeader(key, value)


if __name__ == "__main__":
    # Your LRUCache object will be instantiated and called as such:
    obj = LRUCache(2)
    obj.put(1, 1)
    obj.put(2, 2)
    print(obj.get(1))
    obj.put(3,3)
    print(obj.get(2))
    obj.put(4,4)
    print(obj.get(3))
    print(obj.get(4))
    # ["LRUCache", "put", "put", "put", "put", "get", "get"]
    # [[2], [2, 1], [1, 1], [2, 3], [4, 1], [1], [2]]

