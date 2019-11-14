# For a given source string and a target string, you should output the first index(from 0) of target string in source string.

# If target does not exist in source, just return -1.
# Clarification

# Do I need to implement KMP Algorithm in a real interview?

# Not necessary. When you meet this problem in a real interview, the interviewer may just want to test your basic implementation ability. But make sure you confirm with the interviewer first.

# Example

# Example 1:

# Input: source = "source" ，target = "target"
# Output: -1	
# Explanation: If the source does not contain the target content, return - 1.

# Example 2:

# Input:source = "abcdabcdefg" ，target = "bcd"
# Output: 1	
# Explanation: If the source contains the target content, return the location where the target first appeared in the source.

# Challenge

# O(n2) is acceptable. Can you implement an O(n) algorithm? (hint: KMP)


class Solution:
    """
    @param source: 
    @param target: 
    @return: return the index
    """
    def strStr(self, source, target):
        # Write your code here
        if len(source) < len(target):
            return -1
        elif len(target) == 0:
            return 0
        else:
            res = True
            for i in range(len(source) - len(target) + 1):
                if source[i] == target[0]:
                    for j in range(len(target)):
                        print(source[i+j],target[j])
                        if source[i + j] != target[j]:
                            res = False
                            break
                    if res:
                        return i
                    res = True
            return -1