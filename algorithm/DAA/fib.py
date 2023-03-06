def fib(n):
    if n == 1:
        return [1]
    elif n == 2:
        return [1, 1]
    else:
        fib = [1, 1]
        for i in range(2, n):
            fib.append(fib[i - 2] + fib[i - 1])
        return fib


def fib_2(n):
    if n < 0:
        return None
    if n <= 2:
        return 1
    else:
        return fib_2(n-1) + fib_2(n-2)
print(fib(3))
print(fib_2(3))