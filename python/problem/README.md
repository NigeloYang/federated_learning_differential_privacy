# python中的学习问题
## 常规拷贝、浅拷贝、深拷贝的区别
首先常规拷贝、深拷贝、浅拷贝都是对象的拷贝，都会生成一个看起来相同的对象，他们本质的区别是拷贝出来的对象的地址是否和原对象一样，也就是地址的复制还是值的复制的区别。
1. 常规拷贝：基本的赋值操作，完全共用一个地址，也就是值的拷贝，只要其中一个变化，两者都会发生变化
2. 浅拷贝：主要跟值得元素类型有关，不可变元素两者不共用一个内存地址（地址拷贝），可变元素两者共用一个内存地址（值的拷贝）
3. 深拷贝：所有内容都是地址拷贝，不共用内存地址  

深拷贝和浅拷贝的本质区别时：在浅拷贝时，拷贝出来的新对象的地址和原对象是不一样的，但是新对象里面的可变元素（如列表）的地址和原对象里的可
变元素的地址是相同的，也就是说浅拷贝它拷贝的是浅层次的数据结构（不可变元素），对象里的可变元素作为深层次的数据结构并没有被拷贝到新地址里
面去，而是和原对象里的可变元素指向同一个地址，所以在新对象或原对象里对这个可变元素做修改时，两个对象是同时改变的，但是深拷贝不会这样，这
个是浅拷贝相对于深拷贝最根本的区别。  

不可变元素：int,float,complex,long,str,unicode,tuple
```python
    print('常规拷贝') # chang'gu 
    x = [1, 2, 3]
    y = x
    print(x, y)
    print(id(x), id(y))
    y.append(4)
    y[1] = 11
    print(x, y)
    print(id(x), id(y))
    x.append(5)
    x[1] = 12
    print(x, y)
    print(id(x), id(y))
    
    print('浅拷贝') # 
    a = [1, 2, 3]
    b = a.copy()
    print(a, b)
    print(id(a[0]), id(b[0]))
    print(id(a[3]), id(b[3]))
    a.append(4)
    a[1] = 11
    print(a, b)
    print(id(a[0]), id(b[0]))
    print(id(a[3]), id(b[3]))
    b.append(5)
    b[1] = 12
    print(a, b)
    print(id(a[0]), id(b[0]))
    print(id(a[3]), id(b[3]))
    
    print('浅拷贝失效')
    a1 = [1, 2, 3, [20, 21]]
    b1 = a1.copy()
    print(a1, b1)
    print(id(a1[0]), id(b1[0]))
    print(id(a1[3]), id(b1[3]))
    a1.append(4)
    a1[1] = 11
    a1[3].append(22)
    a1[3][0] = 30
    print(a1, b1)
    print(id(a1[0]), id(b1[0]))
    print(id(a1[3]), id(b1[3]))
    b1.append(5)
    b1[1] = 12
    b1[3].append(23)
    b1[3][0] = 40
    print(a1, b1)
    print(id(a1[0]), id(b1[0]))
    print(id(a1[3]), id(b1[3]))
    
    print('深拷贝')
    a2 = [1, 2, 3, [20, 21]]
    b2 = a2.deepcopy()
    print(a2, b2)
    print(id(a2[0]), id(b2[0]))
    print(id(a2[3]), id(b2[3]))
    a2.append(4)
    a2[1] = 11
    a2[3].append(22)
    a2[3][0] = 30
    print(a2, b2)
    print(id(a2[0]), id(b2[0]))
    print(id(a2[3]), id(b2[3]))
    b2.append(5)
    b2[1] = 12
    b2[3].append(23)
    b2[3][0] = 40
    print(a2, b2)
    print(id(a2[0]), id(b2[0]))
    print(id(a2[3]), id(b2[3]))
```

## 