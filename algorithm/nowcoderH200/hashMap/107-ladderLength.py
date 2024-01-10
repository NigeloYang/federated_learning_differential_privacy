# -*- coding: utf-8 -*-
# @Time    : 2023/11/1
from collections import deque
from typing import List


class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        if endWord not in wordList:
            return 0
        
        wordset = set(wordList)
        queue = [beginWord]
        level = 0
        while queue:
            level += 1
            for i in range(len(queue)):
                word = queue.pop(0)
                wordl = list(word)
                for j in range(len(wordl)):
                    originalword = wordl[j]
                    for ch in range(ord('a'), ord('z') + 1):
                        if chr(ch) == originalword:
                            continue
                        
                        wordl[j] = chr(ch)
                        newword = ''.join(wordl)
                        if newword == endWord:
                            return level + 1
                        
                        if newword in wordset:
                            queue.append(newword)
                            wordset.remove(newword)
                    wordl[j] = originalword
        return 0
    
    def ladderLength2(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        if endWord not in wordList:
            return 0
        
        wordset = set(wordList)
        queue = deque([(beginWord, 1)])
        while queue:
            word, level = queue.popleft()
            for j, val in enumerate(word):
                for ch in range(ord('a'), ord('z') + 1):
                    if chr(ch) == val:
                        continue
                    
                    newword = word[:j] + chr(ch) + word[j + 1:]
                    if newword == endWord:
                        return level + 1
                    
                    if newword in wordset:
                        queue.append((newword, level + 1))
                        wordset.remove(newword)
        return 0


if __name__ == "__main__":
    # for i in range(ord('a'), ord('z') + 1):
    #     print(i)
    #     print(chr(i))
    # for one in range(97, 123):
    #     print(chr(one))
    
    print(Solution().ladderLength("eat", "cow", ["cat", "cot", "eaw", "eaw", "caw"]))
    print(Solution().ladderLength("eat", "cow", ["cat", "cot", "cow", "cet", "ant", "dog"]))
    print(Solution().ladderLength2("eat", "cow", ["cat", "cot", "cow", "cet", "ant", "dog"]))
