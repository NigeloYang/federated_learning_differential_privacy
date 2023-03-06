class ListNode(object):
    # 结点初始化函数, p 即模拟所存放的下一个结点的地址
    # 为了方便传参, 设置 p 的默认值为 0
    def __init__(self, val, next=None):
        self.val = val
        self.next = next


class LinkList(object):
    def __init__(self):
        self.head = None
    
    # 链表初始化函数, 方法类似于尾插
    def initList(self, data):
        # 创建头结点
        self.head = ListNode(data[0])
        p = self.head
        # 逐个为 data 内的数据创建结点, 建立链表
        for i in data[1:]:
            node = ListNode(i)
            p.next = node
            p = p.next
            
    # 创建循环链表
    def initList_c(self, data):
        # 创建头结点
        self.head = ListNode(data[0])
        p = self.head
        # 逐个为 data 内的数据创建结点, 建立链表
        for i in data[1:]:
            node = ListNode(i)
            p.next = node
            p = p.next
        p.next = self.head
    
    # 链表判空
    def isEmpty(self):
        if self.head.next == None:
            print("Empty List!")
            return 1
        else:
            return 0
    
    # 取链表长度
    def getLength(self):
        if self.isEmpty():
            exit(0)
        
        p = self.head
        len = 0
        while p:
            len += 1
            p = p.next
        return len
    
    # 遍历链表
    def traveList(self):
        if self.isEmpty():
            exit(0)
        print('link list traving result: ')
        p = self.head
        while p:
            print(p.val, end='\t')
            p = p.next
        print('')
    
    # 遍历循环链表
    def traveList_c(self):
        if self.isEmpty():
            exit(0)
        print('Link List Traving Result:')
        p = self.head
        while p:
            print(p.val, end='\t')
            if p.next == self.head:
                break
            p = p.next
        print('')
    # 链表插入数据函数
    def insertElem(self, key, index):
        if self.isEmpty():
            exit(0)
        if index < 0 or index > self.getLength() - 1:
            print("Key Error! Program Exit.")
            exit(0)
        
        p = self.head
        i = 0
        while i <= index:
            pre = p
            p = p.next
            i += 1
        
        # 遍历找到索引值为 index 的结点后, 在其后面插入结点
        node = ListNode(key)
        pre.next = node
        node.next = p
    
    # 链表删除数据函数
    def deleteElem(self, index):
        if self.isEmpty():
            exit(0)
        if index < 0 or index > self.getLength() - 1:
            print("Value Error! Program Exit.")
            exit(0)
        
        i = 0
        p = self.head
        # 遍历找到索引值为 index 的结点
        while p.next:
            pre = p
            p = p.next
            i += 1
            if i == index:
                pre.next = p.next
                p = None
                return 1
        
        # p的下一个结点为空说明到了最后一个结点, 删除之即可
        pre.next = None


if __name__ == "__main__":
    # 初始化链表与数据
    data = [1, 2, 3, 4, 5]
    l = LinkList()
    l.initList(data)
    l.traveList()
    print('lenght: ',l.getLength())
    
    # 插入结点到索引值为3之后, 值为666
    l.insertElem(666, 3)
    l.traveList()
    
    # 删除索引值为4的结点
    l.deleteElem(4)
    l.traveList()
