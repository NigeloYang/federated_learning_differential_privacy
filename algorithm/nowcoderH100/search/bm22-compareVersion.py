# -*- coding: utf-8 -*-
# @Time    : 2023/9/6

class Solution:
    def compare(self , version1: str, version2: str) -> int:
        v1 = version1.split('.')
        v2 = version2.split('.')
        print(v1,v2)
        for i in range(max(len(v1),len(v2))):
            ver1 = int(v1[i]) if i < len(v1) else 0
            ver2 = int(v2[i]) if i < len(v2) else 0
            if ver1 > ver2:
                return 1
            elif ver1 < ver2:
                return -1
        return 0

if __name__ == "__main__":
    print(Solution().compare("1.1","2.1"))
