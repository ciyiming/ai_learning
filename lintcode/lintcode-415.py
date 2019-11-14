# Description
# 中文
# English

# Given a string, determine if it is a palindrome, considering only alphanumeric characters and ignoring cases.

# Example

# Example 1:

# Input: "A man, a plan, a canal: Panama"
# Output: true
# Explanation: "amanaplanacanalpanama"

# Example 2:

# Input: "race a car"
# Output: false
# Explanation: "raceacar"

# Challenge

# O(n) time without extra memory.

class Solution:
    """
    @param s: A string
    @return: Whether the string is a valid palindrome
    """
    def isPalindrome(self, s):
        # write your code here
        i, j = 0, len(s)-1
        vl = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890'
        while i <= j-1:
            if vl.find(s.upper()[i]) == -1:
                i += 1
                continue
            if vl.find(s.upper()[j]) == -1:
                j -= 1
                continue
            if s.upper()[i] != s.upper()[j]:
                return False
            else:
                i += 1
                j -= 1
        return True