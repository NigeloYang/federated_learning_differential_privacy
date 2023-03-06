'''两个数组的交集 II

给你两个整数数组 nums1 和 nums2 ，请你以数组形式返回两数组的交集。返回结果中每个元素出现的次数，
应与元素在两个数组中都出现的次数一致（如果出现次数不一致，则考虑取较小值）。可以不考虑输出结果的顺序。


示例 1：
输入：nums1 = [1,2,2,1], nums2 = [2,2]
输出：[2,2]

示例 2:
输入：nums1 = [4,9,5], nums2 = [9,4,9,8,4]
输出：[4,9]
'''


def intersect(nums1, nums2):
    res = {}
    temp = []
    if len(nums1) > len(nums2):
        maxs = nums1
        mins = nums2
    else:
        maxs = nums2
        mins = nums1

    for i in range(len(maxs)):
        if maxs[i] in res:
            res.update({maxs[i]: res.get(maxs[i]) + 1})
        res.setdefault(maxs[i], 1)
    print(res)
    for i in range(len(mins)):
        if mins[i] in res:
            if res.get(mins[i]) > 0:
                res.update({mins[i]: res.get(mins[i]) - 1})
                temp.append(mins[i])
            else:
                del res[mins[i]]

    return temp


print(intersect([1, 2, 2, 1], [2, 2]))
print(intersect([4, 9, 5], [9, 4, 9, 8, 4]))
print(intersect([3, 1, 2], [1, 1]))
