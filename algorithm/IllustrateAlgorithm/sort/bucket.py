# -*- coding: utf-8 -*-
# @Time    : 2023/6/25

'''桶排序

考虑一个长度为 n 的数组，元素是范围[0,1) 的浮点数。桶排序的流程如下：

初始化 k 个桶，将 n 个元素分配到 k 个桶中；
对每个桶分别执行排序（本文采用编程语言的内置排序函数）；
按照桶的从小到大的顺序，合并结果；
'''


class Solution:
    def bucket_sort(self, nums):
        k = len(nums) // 2
        buckets = [[] for _ in range(k)]
        ma = max(nums)
        mi = min(nums)
        # 将数据分配到指定桶中
        for num in nums:
            # 映射索引范围
            i = int(num // 20)
            buckets[i].append(num)
        
        # 桶排序
        for bucket in buckets:
            bucket.sort()
        
        # 合并桶中的结果
        res = []
        for bucket in buckets:
            res.extend(bucket)
        return res


if __name__ == "__main__":
    print(Solution().bucket_sort([49, 99, 82, 9, 57, 43, 91, 75, 15, 37]))
