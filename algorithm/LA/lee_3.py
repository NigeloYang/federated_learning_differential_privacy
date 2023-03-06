'''
lee3 无重复字符的最长子串
给定一个字符串，请你找出其中不含有重复字符的 最长子串 的长度。
输入: s = “abcabcbb”
输出: 3
解释: 因为无重复字符的最长子串是 “abc”，所以其长度为 3。

输入: s = “bbbbb”
输出: 1
解释: 因为无重复字符的最长子串是 “b”，所以其长度为 1。

输入: s = “pwwkew”
输出: 3
解释: 因为无重复字符的最长子串是 “wke”，所以其长度为 3。
请注意，你的答案必须是 子串 的长度，“pwke” 是一个子序列，不是子串。

输入: s = “”
输出: 0
'''


def lengthOfLongestSubstring(s):
    # 哈希集合，记录每个字符是否出现过
    occ = set()
    length = len(s)

    # 右指针，初始值为 0，相当于我们在字符串的左边界，还没有开始移动
    right = res = 0
    for left in range(length):
        if left != 0:
            # 左指针向右移动一格，移除一个字符
            occ.remove(s[left - 1])
        while right < length and s[right] not in occ:
            # 不断地移动右指针
            occ.add(s[right])
            right += 1

        res = max(res, right - left + 1)

    return res


print(lengthOfLongestSubstring('advgbhfds'))

'''
lee 159 至多包含两个不同字符的最长子串
题目：给定一个字符串 s ，找出 至多 包含两个不同字符的最长子串 t ，并返回该子串的长度。

输入: “eceba”
输出: 3
解释: t 是 “ece”，长度为3。

输入: “ccaabbb”
输出: 5
解释: t 是 “aabbb”，长度为5。
'''

'''
lee 209 长度最小的子数组

给定一个含有 n 个正整数的数组和一个正整数 target 。
找出该数组中满足其和 ≥ target 的长度最小的 连续子数组 [numsl, numsl+1, …, numsr-1, numsr] ，并返回其长度。如果不存在符合条件的子数组，返回 0 。

输入：target = 7, nums = [2,3,1,2,4,3]
输出：2
解释：子数组 [4,3] 是该条件下的长度最小的子数组。

输入：target = 4, nums = [1,4,4]
输出：1

输入：target = 11, nums = [1,1,1,1,1,1,1,1]
输出：0

'''


def minSubArrayLen(target, nums):
    length = len(nums)

    # 方案一
    minsize = sums = left = 0

    for right in range(length):
        sums += nums[right]
        while sums >= target:
            minsize = min(minsize, right - left + 1)
            sums -= nums[left]
            left += 1

    # 方案2


    return minsize


print(minSubArrayLen(7, [2, 3, 1, 2, 4, 3]))

'''
lee340

'''
