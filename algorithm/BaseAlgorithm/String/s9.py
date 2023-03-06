'''最长公共前缀
编写一个函数来查找字符串数组中的最长公共前缀。
如果不存在公共前缀，返回空字符串""。

示例 1：
输入：strs = ["flower","flow","flight"]
输出："fl"

示例 2：
输入：strs = ["dog","racecar","car"]
输出：""
解释：输入不存在公共前缀。

提示：
1 <= strs.length <= 200
0 <= strs[i].length <= 200
strs[i] 仅由小写英文字母组成
'''


def longestCommonPrefix(strs):
    print('-------')
    if len(strs) == 1:
        return strs[0]
    # 寻找最小字符串
    min_sl = len(strs[0])
    min_str = strs[0]
    for min_s in strs:
        if len(min_s) < min_sl:
            min_sl = len(min_s)
            min_str = min_s
    print(f"min_sl: {min_sl}  min_str:{min_str}")
    # 查找最小公共前缀
    for i in range(min_sl, 0, -1):
        temp = []
        for item in strs:
            # temp.append(1 if item.find(min_str[:i]) >= 0 else -1)
            temp.append(1 if item[:i] == min_str[:i] else -1)
        if sum(temp) == len(strs):
            return min_str[:i]
    return ''
    
    # while start < hay_len:
    #     if haystack[start] == needle[0]:
    #         if start + ned_len <= hay_len:
    #             if haystack[start:start+ned_len] == needle:
    #                 return start
    #             else:
    #                 start += 1
    #         else:
    #             return -1
    #     else:
    #         start += 1
    # return -1
    



print(longestCommonPrefix(["flower", "flow", "flight"]))
print(longestCommonPrefix(["dog", "racecar", "car"]))
print(longestCommonPrefix(["asdf"]))
print(longestCommonPrefix(["cir","car"]))
print(longestCommonPrefix(["reflower","flow","flight"]))
