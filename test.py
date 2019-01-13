class Solution:
    def isPalindrome(self,st):
        l = len(st)
        for k in range(0, l):
            if not (st[k] == st[l-(k+1)]):
                return False
        return True
        
    
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        pal = ""
        if len(s) == 1:
            return s
        elif (len(s) == 0):
            return pal
        for i in range(0, len(s)):
            for j in range(i, len(s)):
                st = s[i:j]
                if (self.isPalindrome(st)):
                    if (len(st) >= len(pal)):
                        pal = st
        return pal