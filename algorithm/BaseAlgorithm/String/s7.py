'''
给你两个字符串haystack 和 needle，请你在 haystack字符串中找出 needle字符串的第一个匹配项的下标（下标从 0 开始）。
如果needle 不是 haystack 的一部分，则返回-1 。

示例 1：
输入：haystack = "sadbutsad", needle = "sad"
输出：0
解释："sad" 在下标 0 和 6 处匹配。
第一个匹配项的下标是 0 ，所以返回 0 。

示例 2：
输入：haystack = "leetcode", needle = "leeto"
输出：-1
解释："leeto" 没有在 "leetcode" 中出现，所以返回 -1。

提示：
1 <= haystack.length, needle.length <= 104
haystack 和 needle 仅由小写英文字符组成
'''


def strStr(haystack, needle):
    # 方案1
    # return haystack.find(needle)
    
    # 方案2
    hay_len = len(haystack)
    ned_len = len(needle)
    start = 0
    while start < hay_len:
        if haystack[start] == needle[0]:
            if start + ned_len <= hay_len:
                if haystack[start:start+ned_len] == needle:
                    return start
                else:
                    start += 1
            else:
                return -1
        else:
            start += 1
    return -1


print(strStr("sadbutsad", "sad"))
print(strStr("adbutsad", "sad"))
print(strStr("leetcode", "leeto"))
