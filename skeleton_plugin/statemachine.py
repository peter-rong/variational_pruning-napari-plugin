# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 15:21:26 2022

@author: Yigan
"""

class State:
    
    def __init__(self):
        self.parentMachine = None
    
    def on_start(self):
        pass
    
    def execute(self):
        pass
    
    def on_finish(self):
        pass
    
    def get_next(self):
        return None


class StateMachine:
    
    def __init__(self):
        self.current = None
    
    def execute(self):
        if self.valid():
            self.current.execute()
            
    def to_next(self):
        if self.valid():
            self.change_state(self.current.get_next())
    
    def change_state(self, state):
        if self.valid():
            self.current.on_finish()
        self.current = state
        if self.valid():
            self.current.parentMachine = self
            self.current.on_start()
    
    def valid(self) -> bool:
        return self.current is not None