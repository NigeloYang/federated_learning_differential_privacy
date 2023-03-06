'''有效的字母异位词
给定两个字符串 s 和 t ，编写一个函数来判断 t 是否是 s 的字母异位词。

注意：若 s 和 t 中每个字符出现的次数都相同，则称 s 和 t 互为字母异位词。


示例1:

输入: s = "anagram", t = "nagaram"
输出: true
示例 2:

输入: s = "rat", t = "car"
输出: false

提示:
1 <= s.length, t.length <= 5 * 104
s 和 t 仅包含小写字母
'''


def isAnagram(s, t):
    # 方法一，哈希值
    if len(s) != len(t):
        return False

    # temp1 = {}
    # for i in range(len(s)):
    #     if s[i] in temp1:
    #         temp1.update({s[i]: temp1.get(s[i]) + 1})
    #     else:
    #         temp1.setdefault(s[i], 1)
    # for i in range(len(t)):
    #     if t[i] in temp1:
    #         if temp1.get(t[i]) > 0:
    #             temp1.update({t[i]: temp1.get(t[i]) - 1})
    #         else:
    #             return False
    #     else:
    #         return False
    # return True
    
    # 方法2
    # temp1 = s.split()
    # temp2 = t.split()
    temp1 = list(s)
    temp2 = list(t)
    print(temp1)
    print(temp2)
    return sorted(temp1) == sorted(temp2)


print(isAnagram(s="rat", t="cars"))
print(isAnagram(s="anagram", t="nagaram"))
print(isAnagram(s="rat", t="car"))
print(isAnagram(s="aacc", t="ccac"))
