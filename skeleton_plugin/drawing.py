# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 15:01:53 2022

@author: Yigan
"""

class ShapeConfig:
    
    def __init__(self):
        self.face_color = ""
        self.edge_color = ""
        self.size = 0.0


class LayerConfig:
    
    def __init__(self):
        self.name = ""
        self.face_color = ""
        self.edge_color = ""
        self.size = 0.0
        self.opacity = 0.0


class PointEdgeConfig:
    
    def __init__(self, pConfig : LayerConfig, pEdgeConfig : LayerConfig):
        self.pointConfig = pConfig
        self.edgeConfig = pEdgeConfig

def default_config() -> LayerConfig:
    config = LayerConfig();
    config.name = "new_layer"
    config.face_color = "blue"
    config.edge_color = "blue"
    config.size = 1.0
    config.opacity = 0.8
    
    return config



def angular_default_config() ->LayerConfig:
    config = LayerConfig();
    config.name = "new_layer"
    config.face_color = "white"
    config.edge_color = "white"
    config.size = 1.0
    config.opacity = 0.8

    return config