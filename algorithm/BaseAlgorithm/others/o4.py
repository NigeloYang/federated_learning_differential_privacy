'''杨辉三角
给定一个非负整数 numRows，生成「杨辉三角」的前 numRows 行。

在「杨辉三角」中，每个数是它左上方和右上方的数的和。
示例 1:
输入: numRows = 5
输出: [[1],[1,1],[1,2,1],[1,3,3,1],[1,4,6,4,1]]
示例2:
输入: numRows = 1
输出: [[1]]
'''


class Solution:
    def generate(self, numRows: int):
        if numRows == 0:
            return []
        elif numRows == 1:
            return [[1]]
        elif numRows == 2:
            return [[1], [1, 1]]
        else:
            res = [[1], [1, 1]]
            temp = [1, 1]
            for i in range(2, numRows):
                a = [1]
                j = 1
                while j < i:
                    a.append(temp[j-1] + temp[j])
                    j += 1
                a.append(temp[-1])
                res.append(a)
                temp = a
            return res


if __name__ == '__main__':
    # print(Solution().generate(1))
    print(Solution().generate(3))
    print(Solution().generate(4))
    print(Solution().generate(5))
