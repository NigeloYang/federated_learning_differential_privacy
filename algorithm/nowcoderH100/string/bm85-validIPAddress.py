# -*- coding: utf-8 -*-
# @Time    : 2023/9/27

class Solution:
    def validIPAddress(self, IP: str) -> str:
        if not str:
            return 'Neither'
        
        if '.' in IP:
            ip4 = IP.split('.')

            if len(ip4) != 4:
                return 'Neither'
            for ip in ip4:
                if (len(ip) > 1 and ip[0] == '0') or len(ip) > 3:
                    return 'Neither'
                if not ip.isdigit():
                    return 'Neither'
                if int(ip) > 255 or int(ip) < 0:
                    return 'Neither'
            return "IPv4"
        elif ':' in IP:
            ip6 = IP.split(':')
            if len(ip6) != 8:
                return 'Neither'
            for ip in ip6:
                if len(ip) > 4 or (ip == ''):
                    return 'Neither'
                for i in ip:
                    if i < '0' or (i > '9' and i < 'A') or (i > 'F' and i < 'a') or i > 'f':
                        return 'Neither'
            return 'IPv6'
        return 'Neither'
                
            
            
if __name__ == "__main__":
    # print(Solution().validIPAddress('172.16.254.1'))
    # print(Solution().validIPAddress('172.16.254.01'))
    # print(Solution().validIPAddress('172.16.254.0'))
    # print(Solution().validIPAddress('172.16.254.-1'))
    print(Solution().validIPAddress('2001:0db8:85a3:0000:0000:8a2e:0370:7334'))
    print(Solution().validIPAddress('2001:db8:85a3:0:0:8A2E:0370:7334'))
    print(Solution().validIPAddress('2001:db8:85a3::8A2E:0370:7334'))
    print(Solution().validIPAddress('02001:db8:85a3:0:0:8A2E:0370:7334'))
    print(Solution().validIPAddress('02001:db8:85@3:0:0:8A2E:0370:7334'))
    print(Solution().validIPAddress('20EE:FGb8:85a3:0:0:8A2E:0370:7334'))
