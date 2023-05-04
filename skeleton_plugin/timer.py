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

        self.stamps = list()

    def print_solver_time(self):
        print("------ printing solver total time ------")
        total_delta = 0
        total_delta_without_draw = 0
        for stamp in self.stamps:
            st, time, delta = stamp
            if st == "compute angle function and cluster" or st == "PCST_solver" or st =="PCST_to_Graph":
                total_delta += delta
                total_delta_without_draw += delta
            if st == "draw_skeleton_result":
                total_delta += delta
        to_print = "Solver spends a total of  : " +str(total_delta) +" time to run."
        to_print_second = "Without the draw step, it takes  : " +str(total_delta_without_draw) +" time to run."
        print(to_print)
        print(to_print_second)
        print("------ end of computing solver total time ------")

    def clear(self):
        self.stamps.clear()
    
    def __stamp_to_str(self, stamp):
        st,time,delta = stamp
        return "stamp : " + st + ", current : " + str(time) + ", delta : " + str(delta)

    def stamp_to_time(self,stamp):
        st, time, delta = stamp
        return time