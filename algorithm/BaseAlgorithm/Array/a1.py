'''删除数组中的重复项

示例 1：

输入：nums = [1,1,2]
输出：2, nums = [1,2,_]
解释：函数应该返回新的长度 2 ，并且原数组 nums 的前两个元素被修改为 1, 2 。不需要考虑数组中超出新长度后面的元素。
示例 2：

输入：nums = [0,0,1,1,1,2,2,3,3,4]
输出：5, nums = [0,1,2,3,4]
解释：函数应该返回新的长度 5 ， 并且原数组 nums 的前五个元素被修改为 0, 1, 2, 3, 4 。不需要考虑数组中超出新长度后面的元素。

相似题目
数组8: 移除元素的双指针优化, lee 27
数组9: 删除重复元素的通解问题, lee 26,80
'''


def removeDuplicates_1(nums):
    if len(nums) <= 1:
        return len(nums)

    n = len(nums)
    left, right = 0, 1
    while right < n:
        if nums[left] != nums[right]:
            left += 1
            nums[left] = nums[right]
        right += 1
    return left + 1

print(removeDuplicates_1([1]))
print(removeDuplicates_1([1, 2]))
print(removeDuplicates_1([1, 1, 2]))
print(removeDuplicates_1([0, 0, 1, 1, 1, 2, 2, 3, 3, 4]))

def removeDuplicates_2(nums):
    # 在数组上进行迭代删除
    for i in range(len(nums) - 1, 0, -1):
        if nums[i] == nums[i - 1]:
            # del nums[i]
            nums.pop(i)

    # for i in range(0, len(nums) - 1):
    #     if i + 1 == len(nums) - 1:
    #         if nums[i] == nums[i + 1]:
    #             return len(nums.pop(i + 1))
    #     elif nums[i] == nums[i + 1]:
    #         nums.pop(i)
    return len(nums)


print(removeDuplicates_2([1]))
print(removeDuplicates_2([1, 2]))
print(removeDuplicates_2([1, 1, 2]))
print(removeDuplicates_2([0, 0, 1, 1, 1, 2, 2, 3, 3, 4]))
