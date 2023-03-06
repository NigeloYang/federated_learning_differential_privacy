'''移动零
给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序。
请注意，必须在不复制数组的情况下原地对数组进行操作。

示例 1:
输入: nums = [0,1,0,3,12]
输出: [1,3,12,0,0]

示例 2:
输入: nums = [0]
输出: [0]

提示:
1 <= nums.length <= 10^4
-2^31<= nums[i] <= 2^31- 1
'''


def moveZeroes(nums):
    """
    Do not return anything, modify nums in-place instead.
    """
    # 排序也可以，但是提交有变化
    # num = sorted(nums, reverse=True)
    # idx = nums.index(0)
    # num = sorted(num[:idx]) + num[idx:]
    # print(num)

    # 指针
    temp1, temp2 = [], []
    # for i in nums:
    #     if i == 0:
    #         temp.append(i)
    #         nums.remove(i)

    # for i in range(len(nums)):
    #     if nums[i] == 0:
    #         temp1.append(nums[i])
    #     else:
    #         temp2.append(nums[i])
    # res = []
    # res.extend(temp2)
    # res.extend(temp1)
    # nums = res
    # nums = temp2 + temp1
    for i in nums:
        if i == 0:
            nums.remove(i)
            nums.append(0)
    print(nums)


moveZeroes([0, 1, 0, 3, 12])
moveZeroes([0])
moveZeroes([0, 0, 1])
