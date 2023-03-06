# 最优时间复杂度O（nlogn）最坏时间复杂度O（n^2）
def quicksort(arr):
    if len(arr) < 2:
        return arr
    else:
        pivots = arr[0]
        less = [i for i in arr[1:] if i <= pivots]
        greater = [i for i in arr[1:] if i > pivots]

        return quicksort(less) + [pivots] + quicksort(greater)


arr = [3, 5, 8, 2, 67, 24]
print(quicksort(arr))
