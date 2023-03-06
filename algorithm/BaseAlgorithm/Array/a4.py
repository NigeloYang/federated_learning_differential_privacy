'''存在重复元素
给你一个整数数组 nums 。如果任一值在数组中出现 至少两次 ，返回 true ；如果数组中每个元素互不相同，返回 false 。

示例 1：
输入：nums = [1,2,3,1]
输出：true

示例 2：
输入：nums = [1,2,3,4]
输出：false

示例3：
输入：nums = [1,1,1,3,3,4,3,2,4,2]
输出：true
'''


def containsDuplicate(nums):
    # 比较长度
    # if len(nums) != len(set(nums)): return True
    # return False

    # 重新排序比较前后是否相同
    # nums = sorted(nums)
    # for i in range(1, len(nums)):
    #     if nums[i - 1] == nums[i]:
    #         return True
    # return False

    # 创建hashmap
    arr = {}
    for i in range(len(nums)):
        if nums[i] in arr:
           return True
        arr.setdefault(nums[i], 1)
    return False

print(containsDuplicate([1, 2, 3, 1]))
print(containsDuplicate([1, 2, 3, 4]))
print(containsDuplicate([1, 1, 1, 3, 3, 4, 3, 2, 4, 2]))
