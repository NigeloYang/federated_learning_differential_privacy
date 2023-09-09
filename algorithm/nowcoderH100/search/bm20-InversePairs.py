# -*- coding: utf-8 -*-
# @Time    : 2023/9/6

class Solution:
    def InversePairs(self, nums: List[int]) -> int:
        count = 0
        
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                if nums[i] > nums[j]:
                    count += 1
        count = count % 1000000007
        return count

class Solution2:
    mod = 1000000007

    def InversePairs(self, nums: List[int]) -> int:
        n = len(nums)
        res = [0] * n

        return self.mergeSort(0, n-1, nums, res)

    def mergeSort(self, left:int, right:int, data: List[int], temp: List[int]):
        if left >= right:
            return 0

        mid = int((left + right) / 2)
        res = self.mergeSort(left, mid, data, temp) + self.mergeSort(mid + 1, right, data, temp)

        res %= self.mod
        l, r = left, mid + 1

        for k in range(left, right+1):
            temp[k] = data[k]

        # 合并
        for k in range(left, right+1):
            # 左数组已经合并完成
            if l > mid + 1:
                data[k] = temp[r]
                r += 1
            elif r >= right + 1 or temp[l] <= temp[r]:
                data[k] = temp[l]
                l += 1
            # 左边比右边大，答案增加
            else:
                data[k] = temp[r]
                r += 1
                # 统计逆序对
                res += mid - l + 1
        return res % self.mod
    
if __name__ == "__main__":
    print()
