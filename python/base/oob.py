class Student(object):
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return 'Student Object Name: %s' % self.name


print(Student('Nigelo'))


class Fib(object):
    def __init__(self):
        self.a, self.b = 0, 1

    def __iter__(self):
        return self

    def __next__(self):
        self.a, self.b = self.b, self.a + self.b
        if self.a > 100:
            raise StopIteration()
        return self.a, self.b


for n in Fib():
    print(n)


class Fibs(object):
    def __init__(self):
        self.a, self.b = 0, 1

    def __iter__(self):
        return self

    def __next__(self):
        self.a, self.b = self.b, self.a + self.b
        if self.a > 100:
            raise StopIteration()
        return self.a, self.b

    def __getitem__(self, n):
        a, b = 1, 1
        for x in range(n):
            a, b = b, a + b
        return a, b


for x in range(10):
    print(Fibs()[x])
