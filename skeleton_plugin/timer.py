# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 14:02:01 2022

@author: Yigan
"""

import time

class TimeRecord:

    def __init__(self):
        self.stamps = list()
    
    def count(self):
        return len(self.stamps)
    
    def stamp(self, name : str):
        current = time.time()
        num = self.count()
        delta = 0 if num <= 0 else current - self.stamps[num-1][1]
        self.stamps.append((name,current,delta))
    
    def print_records(self):
        print("------ printing time stamp records ------")
        for stamp in self.stamps:
            print(self.__stamp_to_str(stamp))
        print("------ end of records ------")
    
    def clear(self):
        self.stamps.clear()
    
    def __stamp_to_str(self, stamp):
        st,time,delta = stamp
        return "stamp : " + st + ", current : " + str(time) + ", delta : " + str(delta)
 