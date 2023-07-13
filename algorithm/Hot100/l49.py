# -*- coding: utf-8 -*-
# @Time    : 2023/7/3
'''字母异位词分组
给你一个字符串数组，请你将 字母异位词 组合在一起。可以按任意顺序返回结果列表。

字母异位词 是由重新排列源单词的所有字母得到的一个新单词。
'''
import collections


class Solution:
    def groupAnagrams(self, strs):
        res = collections.defaultdict(list)
        for st in strs:
            key = "".join(sorted(st))
            res[key].append(st)
        return list(res.values())
    
if __name__ == "__main__":
    print(Solution().groupAnagrams(["eat", "tea", "tan", "ate", "nat", "bat"]))
