'''
给你一个数组，将数组中的元素向右轮转 k 个位置，其中 k 是非负数

示例 1:
输入: nums = [1,2,3,4,5,6,7], k = 3
输出: [5,6,7,1,2,3,4]
解释:
向右轮转 1 步: [7,1,2,3,4,5,6]
向右轮转 2 步: [6,7,1,2,3,4,5]
向右轮转 3 步: [5,6,7,1,2,3,4]

示例2:
输入：nums = [-1,-100,3,99], k = 2
输出：[3,99,-1,-100]
解释:
向右轮转 1 步: [99,-1,-100,3]
向右轮转 2 步: [3,99,-1,-100]

提示：
1 <= nums.length <= 105
-2^31 <= nums[i] <= 231 - 1
0 <= k <= 105
'''
import copy


def rotate(list, k):
    # 创建临时空数组，进行迭代插入
    # temp = []
    # temp.append(list[k:len(list)])

    # 两次反转
    # list.reverse()
    # print(list)
    # print(list[:k].reverse())
    # print(list[k:])
    # list.reverse()
    # list.reverse(list[k:])

    # 浅拷贝
    # print(list)
    # print(list[k:])
    # print(list[:k])
    # if len(list) <= k:
    #     list.reverse()
    # list[:] = list[-k:] + list[:-k]
    # print(list[-k:])
    # print(list[:-k])
    # list[:] = list[-k:] + list[:-k]

    temp = copy.deepcopy(list)
    for i in range(len(list)):
        list[(i + k) % len(list)] = temp[i]
    return list


nums1 = [1, 2, 3, 4, 5, 6, 7]
nums2 = [-1, -100, 3, 99]
nums = [1, 2]

print(rotate(nums1, 3))
print(rotate(nums2, 2))
print(rotate(nums, 4))
