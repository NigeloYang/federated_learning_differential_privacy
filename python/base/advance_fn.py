from functools import reduce


# 学习Map（f, para） 第一个参数是函数对象，第二个对象是要处理参数类型为 Iterable,参数传递可以有一个或者两个
def var(x):
    return x * x


def add1(x):
    return x + x


r1 = map(var, [1, 2, 3, 4, 5, 6, 7, 8, 9])
r2 = map(add1, [1, 2, 3, 4, 5, 6, 7, 8, 9])
r3 = list(map(str, [1, 2, 3, 4, 5, 6, 7, 8, 9]))
print(list(r1))
print(list(r2))
print(list(r3))


# 学习Reduce（f, para） 第一个参数是函数对象，第二个对象是要处理参数,该函数必须接受两个参数并且它把结果继续和序列的下一个元素做累积计算'''
def add2(x, y):
    return x + y


def fn(x, y):
    return x * 10 + y


r4 = reduce(add2, [1, 2, 3, 4, 5, 6, 7, 8, 9])
r5 = reduce(fn, [1, 2, 3, 4, 5, 6, 7, 8, 9])
print(r4)
print(r5)


def charm2num(s):
    digist = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}
    return digist[s]


print(reduce(fn, map(charm2num, '12345')))

DIGIST = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}


def str2num(s):
    def fn(x, y):
        return x * 10 + y

    def char2num(s):
        return DIGIST[s]

    return reduce(fn, map(char2num, s))


print(str2num('1234567'))


# filter()也接收一个函数和一个序列。
# 和map()不同的是，filter()把传入的函数依次作用于每个元素，然后根据返回值是True还是False决定保留还是丢弃该元素
def isodd(n):
    return n % 2 == 1


print(list(filter(isodd, [1, 2, 3, 4, 5, 6, 7, 8, 9])))


def notempty(s):
    return s and s.strip()


print(list(filter(notempty, ['a', 'B', None, 'c', ' '])))

# 关键字lambda表示匿名函数，冒号前面的x表示函数参数。
# 匿名函数有个限制，就是只能有一个表达式，不用写return，返回值就是该表达式的结果
print(list(map(lambda x: x * x, [1, 2, 3, 4, 5, 6, 7])))

def build(x,y):
    return lambda: x * x  + y * y

f = build(5,4)
print(f())