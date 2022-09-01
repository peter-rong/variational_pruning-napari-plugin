# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 15:46:23 2022

@author: Yigan
"""

import napari
from . import graph
from . import drawing

boundary = "boundary"
voronoi = "voronoi"
internalVoronoi = "internal voronoi"
heatmap = "heatmap"
burnTime = "burn time"
erosionT = "ET"
final = "final"
angle = "angle"
pcst = "pcst" #pr
pcstResult = "pcst result"
skeletonResult = "skeleton result"


class DisplayConfig:
    
    def __init__(self):
        self.show_edgepoints = False
        self.show_voronoi = False
        self.show_internal_voronoi = False
        self.show_heatmap = False
        self.show_bt = False
        self.show_et = False
        self.show_final = True
        self.show_angle = False
        self.show_pcst = False
        self.show_pcst_result = False
        self.show_skeleton_result = False
    
    def flag_raise(self, name : str) -> bool:
        if name == boundary:
            return self.show_edgepoints
        if name == voronoi:
            return self.show_voronoi
        if name == internalVoronoi:
            return self.show_internal_voronoi
        if name == heatmap:
            return self.show_heatmap
        if name == burnTime:
            return self.show_bt
        if name == erosionT:
            return self.show_et
        if name == final:
            return self.show_final
        if name == angle:
            return self.show_angle
        if name == pcst:
            return self.show_pcst
        if name == pcstResult:
            return self.show_pcst_result
        if name == skeletonResult:
            return self.show_skeleton_result
        return False
        

class Display:
    
    current_display = None
    
    def current(): 
        if Display.current_display is None:
            Display.current_display = Display(napari.current_viewer())
        return Display.current_display
    
    def __init__(self, viewer : napari.Viewer):
        self.viewer = viewer
        self.layers = list()
        self.config = DisplayConfig()
    
    def set_config(self, con : DisplayConfig):
        self.config = con

    def draw_layer(self, g : graph.Graph, config : drawing.PointEdgeConfig, name : str) :
        if self.config.flag_raise(name): 
            graph_layer = self.find(name)
            if graph_layer is None:
                graph_layer = GraphLayer.create(name)
            graph_layer.draw(g,config)
            self.layers.append(graph_layer)
    
    def find(self, layer : str):
        for l in self.layers:
            if l.name == layer:
                return l
        return None
    
    def show_layer(self, isShow : bool, layer : str):
        target = self.find(layer)
        if target is not None:
            target.show(isShow)
    
    def reset(self):
        self.show_layer(isShow = self.config.show_edgepoints, layer = boundary)
        self.show_layer(isShow = self.config.show_voronoi, layer = voronoi)
        self.show_layer(isShow = self.config.show_internal_voronoi, layer = internalVoronoi)
        self.show_layer(isShow = self.config.show_heatmap, layer = heatmap)
        self.show_layer(isShow = self.config.show_bt, layer = burnTime)
        self.show_layer(isShow = self.config.show_final, layer = final)
        self.show_layer(isShow = self.config.show_pcst, layer = pcst)
        self.show_layer(isShow = self.config.show_pcst_result, layer = pcstResult)

    def removeall(self):
        # todo : remove all layers
        for l in self.layers:
            l.remove()
        self.layers.clear()

    
    

class GraphLayer:
    
    def __init__(self, name : str, pl : napari.layers.Points, el : napari.layers.Shapes):
        self.name = name
        self.pointLayer = pl
        self.edgeLayer = el
    
    def show(self, isShow : bool):
        self.pointLayer.visible = isShow
        self.edgeLayer.visible = isShow
    
    def remove(self):
        viewer = Display.current().viewer
        if self.pointLayer in viewer.layers:
            viewer.layers.remove(self.pointLayer)
        if self.edgeLayer in viewer.layers:
            viewer.layers.remove(self.edgeLayer)        
    
    def draw(self, g : graph.Graph, config : drawing.PointEdgeConfig):
        pc = config.pointConfig
        ec = config.edgeConfig
        
        
        self.pointLayer.data = g.points
        self.pointLayer.size = pc.size
        if self.pointLayer.visible:
            self.pointLayer.opacity = pc.opacity
            self.pointLayer.face_color = pc.face_color
            self.pointLayer.edge_color = pc.edge_color
        self.pointLayer.selected_data = set()
        
        self.edgeLayer.shape_type = 'line'
        self.edgeLayer.data = g.get_edge_cord()       
        self.edgeLayer.edge_width = ec.size
        if self.edgeLayer.visible:
            self.edgeLayer.edge_color = ec.edge_color
            self.edgeLayer.face_color = ec.face_color
        self.edgeLayer.selected_data = set()
        #self.pointLayer.refresh()
    
    def create(name : str):
        viewer = Display.current().viewer
        '''
        pc = config.pointConfig
        pname = name + " : " + "points"
        pointLayer = viewer.add_points(g.points, name = pname, size = pc.size, opacity = pc.opacity, face_color = pc.face_color, edge_color = pc.edge_color)
    
        ec = config.edgeConfig
        ename = name + " : " + "edges"
        
        shapeLayer = napari.layers.Shapes(name = ename)
        shapeLayer.add_lines(g.get_edge_cord(), edge_width = ec.size, face_color = ec.face_color, edge_color = ec.edge_color)
        viewer.add_layer(shapeLayer)
        '''
        pname = name + " : " + "points"
        ename = name + " : " + "edges"
        pointLayer = napari.layers.Points(name = pname)
        shapeLayer = napari.layers.Shapes(name = ename)
        viewer.add_layer(pointLayer)
        viewer.add_layer(shapeLayer)
        return GraphLayer(name = name, pl = pointLayer, el = shapeLayer)


