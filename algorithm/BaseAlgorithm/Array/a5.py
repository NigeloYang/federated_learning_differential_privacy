'''只出现一次的数字
给定一个非空整数数组，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素。
说明：
你的算法应该具有线性时间复杂度。 你可以不使用额外空间来实现吗？

示例 1:
输入: [2,2,1]
输出: 1

示例2:
输入: [4,1,2,1,2]
输出: 4
'''


def singleNumber(nums):
    # 异或运算
    # res = 0
    # for i in range(len(nums)):
    #     res ^= nums[i]
    # return res

    # 集合
    if len(nums) <= 1:
        return nums[0]

    arr = {}
    for i in range(len(nums)):
        if nums[i] in arr:
            arr.update({nums[i]: arr.get(nums[i]) + 1})
        arr.setdefault(nums[i], 1)
    for (key, val) in arr.items():
        if val == 1:
            return key


print(singleNumber([2, 2, 1]))
print(singleNumber([2, 2, 1, 1]))
print(singleNumber([4, 1, 2, 1, 2]))
