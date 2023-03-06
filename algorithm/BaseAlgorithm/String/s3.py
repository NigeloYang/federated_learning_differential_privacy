'''字符串中的第一个唯一字符
给定一个字符串 s ，找到 它的第一个不重复的字符，并返回它的索引 。如果不存在，则返回 -1 。

示例 1：
输入: s = "leetcode"
输出: 0

示例 2:
输入: s = "loveleetcode"
输出: 2

示例 3:
输入: s = "aabb"
输出: -1
提示:

1 <= s.length <= 105
s 只包含小写字母
'''


def firstUniqChar(s):
    # 方法1 双指针
    length = len(s)
    #
    # if length <= 1:
    #     return 0

    # left,right = 1
    # for i in range(length):
    #     if s[left - 1] == s[right]:

    # 方法2 字典统计 只考虑的偶数出现次数没有考虑到奇数
    # dic = {}
    # for i in range(length):
    #     if s[i] in dic:
    #         del dic[s[i]]
    #     else:
    #         dic.setdefault(s[i], i)
    # if len(dic) == 0:
    #     return -1
    # else:
    #     res = list(dic.values())
    #     return res[0]

    # 3
    dic = {}
    for i in range(len(s)):
        if s[i] in dic:
            dic.update({s[i]: dic.get(s[i]) + 1})
        else:
            dic.setdefault(s[i], 1)

    for key, value in dic.items():
        if value == 1:
            return str.rfind(s, key)
    return -1


print(firstUniqChar("leetcode"))
print(firstUniqChar("loveleetcode"))
print(firstUniqChar("aabb"))
