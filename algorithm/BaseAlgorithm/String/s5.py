'''验证回文串
给定一个字符串，验证它是否是回文串，只考虑字母和数字字符，可以忽略字母的大小写。

说明：本题中，我们将空字符串定义为有效的回文串。

 

示例 1:

输入: "A man, a plan, a canal: Panama"
输出: true
解释："amanaplanacanalpanama" 是回文串
示例 2:

输入: "race a car"
输出: false
解释："raceacar" 不是回文串
'''
import re


def isPalindrome(s):
    if len(s) <= 0:
        return True

    # 左右双指针
    # left = 0
    # right = len(s) - 1
    # while left < right:
    #     if not s[left].isdigit():
    #         if not s[left].isalpha():
    #             left += 1
    #     if not s[right].isdigit():
    #         if not s[right].isalpha():
    #             right -= 1
    #     if s[left].isdigit() and s[right].isdigit():
    #         if not (s[left] == s[right]):
    #             return False
    #     elif s[left].isdigit() and s[right].isalpha():
    #         return False
    #     elif s[left].isalpha() and s[right].isalpha():
    #         if not (s[left].lower() == s[right].lower()):
    #             return False
    # return True

    # 字符
    # res = ''
    # for i in range(len(s)):
    #     if s[i].isdigit():
    #         res += s[i]
    #     elif s[i].isalpha():
    #         res += s[i].lower()
    # if res == res[::-1]:
    #     return True
    # return False


    # 正则匹配
    pattern = r'[a-zA-Z0-9]'
    res = re.findall(pattern, s, re.ASCII)
    return "".join(res).lower() == "".join(res[::-1]).lower()


print(isPalindrome("A man, a plan, a canal: Panama"))
print(isPalindrome("12 A man, a plan, a canal: Panama 21"))
print(isPalindrome("race a car"))
print(isPalindrome("12 race a car 21"))
