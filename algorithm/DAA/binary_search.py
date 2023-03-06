# 二分查找 时间: O(log(n))
def binary_search(list, item):
    low = 0
    high = len(list) - 1

    if low == high:
        return low

    while low < high:
        mid = int((low + high) / 2)
        guess = list[mid]
        if guess == item:
            return mid
        if guess > item:
            high = mid - 1
        else:
            low = mid + 1
    return None


my_list = [1, 3, 5, 7, 9, 28]
print(binary_search(my_list, 5))
print(binary_search(my_list, 3))
