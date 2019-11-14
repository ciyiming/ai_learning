# 200. Longest Palindromic Substring
# Given a string S, find the longest palindromic substring in S. You may assume that the maximum length of S is 1000, and there exists one unique longest palindromic substring.
# Example
# 
# Example 1:
# 
# Input:"abcdzdcab"
# Output:"cdzdc"
# 
# Example 2:
# 
# Input:"aba"
# Output:"aba"
# 
# Challenge
# 
# O(n2) time is acceptable. Can you do it in O(n) time.

class Solution:
    """
    @param s: input string
    @return: the longest palindromic substring
    """
    def longestPalindrome(self, s):
        #  write your code here
        if len(s) <= 1:
            return s
        p_len = 1
        p_s, p_e = 0, 0
        for i in range(len(s)-1):
            if s[i] == s[i+1]:
                for j in range(min(i+1, len(s)-i-1)):
                    if s[i-j] == s[i+1+j]:
                        if p_len < 2 * j + 2:
                            p_len = 2 * j + 2
                            p_s, p_e = i - j, i + 1 + j
                    else:
                        break
            if i < len(s) - 2 and s[i] == s[i+2]:
                for j in range(min(i+1, len(s)-i-2)):
                    if s[i-j] == s[i+2+j]:
                        if p_len < 2 * j + 3:
                            p_len = 2 * j + 3
                            p_s, p_e = i - j, i + 2 + j
                    else:
                        break
        return s[p_s:p_e+1]