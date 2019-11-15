# 585. Maximum Number in Mountain Sequence
# 中文
# English

# Given a mountain sequence of n integers which increase firstly and then decrease, find the mountain top.
# Example

# Example 1:

# Input: nums = [1, 2, 4, 8, 6, 3] 
# Output: 8

# Example 2:

# Input: nums = [10, 9, 8, 7], 
# Output: 10

# Notice

# Arrays are strictly incremented, strictly decreasing


class Solution:
    """
    @param nums: a mountain sequence which increase firstly and then decrease
    @return: then mountain top
    """
    def mountainSequence(self, nums):
        # write your code here
        if len(nums) == 0:
            pass
        elif len(nums) == 1:
            return nums[0]
        elif len(nums) == 2:
            return max(nums[0], nums[1])
        else:
            s, m, e = 0, len(nums)//2, len(nums)-1
            while s < m < e:
                if nums[m] < nums[s] and nums[m] > nums[e]:
                    e = m
                    m = (e-s+1)//2
                elif nums[m] > nums[s] and nums[m] < nums[e]:
                    s = m
                    m = (e-s+1)//2
                else:
                    s += 1
                    e -= 1
            return self.mountainSequence(nums[s:e+1])