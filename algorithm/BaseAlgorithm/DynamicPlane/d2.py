class Solution:
    def maxProfit(self, prices):
        # 方法会超时
        # if len(prices) == 0:
        #     return 0
        # res = 0
        # for i in range(len(prices)):
        #     for j in range(i + 1, len(prices), 1):
        #         res = max(res, prices[j] - prices[i])
        # return res
        
        
        minp = int(1e9)
        maxp = 0
        for price in prices:
            maxp = max(price - minp, maxp)
            minp = min(price,minp)
        return maxp
        


print(Solution().maxProfit([7, 1, 5, 3, 6, 4]))
print(Solution().maxProfit([7, 6, 4, 3, 1]))
