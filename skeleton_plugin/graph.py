# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 15:36:07 2022

@author: Yigan
"""
import math
import random
import numpy as np
from scipy.signal import convolve2d
from scipy.spatial import Voronoi
from matplotlib import cm
from matplotlib import colors as cl

def get_color_value(color):
    return np.average(color[0:2])

def dist2D(p1,p2):
    return math.sqrt(math.pow(p1[0]-p2[0],2) + math.pow(p1[1]-p2[1],2))

def getCentroid(path_list):

    point_list = []

    for p in path_list:
        point_list.append(p.one.point)
        point_list.append(p.other.point)

    x = 0
    y = 0
    for i in point_list:
        x += i[0]
        y += i[1]
    x = x / (len(point_list))
    y = y / (len(point_list))
    return [x,y]

class BinaryImage:
    
    """
    rawData : in the form of a matrix of [r,g,b,a] value
    """
    def __init__(self, rawData : np.ndarray, thresh : int):
        
        numRow,numCol,colorSize = rawData.shape;
        
        self.data = np.zeros((numRow,numCol))
        flags = rawData[:,:,0] > thresh
        self.data[flags] = 1
        '''
        for r in range(numRow):
            for c in range(numCol):
                color_value = get_color_value(rawData[r,c])
                self.data[r,c] = 0 if color_value < thresh else 1
        '''
    
    def position_is_bright(self, x : float, y : float) -> bool:
        xint = int(x)
        yint = int(y)
        numRow,numCol = self.data.shape;
        return xint >= 0 and yint >= 0 and xint < numRow and y < numCol and self.data[xint,yint] == 1

class Graph:
    
    def __init__(self, point_list : list, edge : list, point_ids : list = None, edge_ids : list = None):
        self.points = point_list
        self.edgeIndex = edge
        self.point_ids = self.__build_point_ids(point_ids)
        self.edge_ids = self.__build_edges_ids(edge_ids)


    def get_cord_as_tuples(self):
        edges = list()
        for e in self.edgeIndex:
            x = e[0]
            y = e[1]
            if x >= 0 and y >= 0:
                edges.append(tuple([tuple(self.points[x]), tuple(self.points[y])]))
        return edges

    def get_edge_cord(self) -> np.ndarray:
        edges = list()
        for e in self.edgeIndex:
            x = e[0]
            y = e[1]
            if x >= 0 and y >= 0:
                edges.append([self.points[x],self.points[y]])
        return np.array(edges)
    
    def __build_edges_ids(self, eds : list):
        if eds is None or len(eds) != len(self.edgeIndex):
            return list(range(0,len(self.edgeIndex)))
        else:
            return eds
        

    def __build_point_ids(self, ids : list):
        if ids is None or len(ids) != len(self.points):
            return list(range(0,len(self.points)))
        else:
            return ids


def cluster_to_skeleton(graph:Graph, point_map, point_pair_map) -> Graph:
    new_points = list()
    new_edges = list()

    for point in graph.points:
        point_tuple = tuple(point)

        if point_tuple not in point_map:
            print('wrong')
            break

        for point_pair in point_map[point_tuple]:
            point_pair = [list(point_pair[0]), list(point_pair[1])]
            if point_pair[0] not in new_points:
                new_points.append(point_pair[0])
            if point_pair[1] not in new_points:
                new_points.append(point_pair[1])

            new_edges.append([new_points.index(point_pair[0]), new_points.index(point_pair[1])])


    for edge_cord in graph.get_cord_as_tuples():

        for point_pair in point_pair_map[edge_cord]:
            point_pair = [list(point_pair[0]),list(point_pair[1])]
            if point_pair[0] not in new_points:
                new_points.append(point_pair[0])
            if point_pair[1] not in new_points:
                new_points.append(point_pair[1])

            new_edges.append([new_points.index(point_pair[0]),new_points.index(point_pair[1])])

    return Graph(new_points,new_edges)

class VoronoiDiagram:
    
    def __init__(self, point_list : list):
        self.vor = Voronoi(point_list)
        self.vert_to_regions = self.__build_vert_to_region()
        self.region_to_site = self.__build_region_to_site()
        self.graph = Graph(self.vor.vertices, self.vor.ridge_vertices)
        
    
    def closest_site(self, vertex_id : int) -> (int, float):
        if vertex_id < 0 or vertex_id >= len(self.vor.vertices):
            return ([],-1)
        
        vertex = self.vor.vertices[vertex_id]       
        regions = self.vert_to_regions[vertex_id]
        
        dist = float("inf")
        cursite = 0
        for r in regions:
            siteid = self.region_to_site[r]
            site = self.vor.points[siteid]
            curdist = dist2D(site, vertex)
            if curdist < dist:
                dist = curdist
                cursite = siteid
        
        return (cursite,dist) 
    
    def edge_angle(self, eid : int) -> float:
        edge = self.vor.ridge_vertices[eid]
        points = self.vor.ridge_points[eid]
        p1 = self.vor.points[points[0]]
        p2 = self.vor.points[points[1]]
        
        v1 = self.vor.vertices[edge[0]]
        v2 = self.vor.vertices[edge[1]]
        center = np.mean(np.array([v1,v2]), axis = 0)
        
        
        norm1 = (p1-center)/np.linalg.norm(p1-center)
        norm2 = (p2-center)/np.linalg.norm(p2-center)
        dot = np.dot(norm1,norm2)
        if dot > 1:
            dot = 1
        if dot < -1:
            dot = -1
        angle = np.arccos(dot)
        return angle
        
        
    
    def __build_vert_to_region(self):
        
        dic = dict()
        for rid in range(len(self.vor.regions)):
            region = self.vor.regions[rid]
            for vid in region:
                if vid >= 0:
                    if vid not in dic:
                        dic[vid] = list()
                    dic[vid].append(rid)
        return dic
    
    def __build_region_to_site(self):
        
        dic = dict()
        for pid in range(len(self.vor.point_region)):
            rid = self.vor.point_region[pid]
            if rid >= 0:
                dic[rid] = pid
        return dic


def get_edge_vertices(img : BinaryImage):
    
    filt = np.array([[1,1],[1,1]])
    con = convolve2d(img.data,filt,mode='same')
    tf = con >= 4
    con[tf] = 0
    ids = np.where(con > 0)
    idcomp = np.transpose(np.array([ids[0],ids[1]]))
    return idcomp + [-0.5,-0.5]
    '''
    s = set()
    numRow, numCol = img.data.shape
    for r in range(numRow):
        for c in range(numCol):
            if(img.data[r,c] == 1):
                neighbors = [(r-1,c),(r+1,c), (r,c+1), (r,c-1)]
                for nr,nc in neighbors:
                    if(nr < 0 or nr >= numRow or nc < 0 or nc >= numCol or img.data[nr,nc] == 0):
                         if(r == nr):
                            vc = float(abs(c+nc))/2
                            s.add((r-0.5,vc))
                            s.add((r+0.5,vc))
                         else:
                            vr = float(abs(r+nr))/2
                            s.add((vr,c-0.5))
                            s.add((vr,c+0.5)) 
                
    return list(s)
    '''      

def get_voronoi(points : list) -> VoronoiDiagram:
    return VoronoiDiagram(points)


def prune_graph(graph : Graph, flags : list) -> Graph:
    new_points = list()
    new_ids = list()
    prune_index = [-1] * len(graph.points)
    numPruned = 0
    for i in range(len(flags)):
        if flags[i] == 1 :
            prune_index[i] = numPruned
            new_points.append(graph.points[i])
            new_ids.append(graph.point_ids[i])
        else:
            numPruned += 1
    
    new_edges = list()
    edge_ids = list()
    for j in range(len(graph.edgeIndex)):
        e = graph.edgeIndex[j]
    #for e in graph.edgeIndex:
        ex = e[0]
        ey = e[1]
        if ex >= 0 and ey >= 0:            
            if prune_index[ex] >= 0 and prune_index[ey] >= 0: 
                ex -= prune_index[ex]
                ey -= prune_index[ey]
                new_edges.append([ex,ey])
                edge_ids.append(graph.edge_ids[j])
                
    return Graph(new_points, new_edges, new_ids, edge_ids)

'''
pruned graph
'''
def graph_in_image(vor : Graph, img : BinaryImage) -> Graph:
    flags = [0]*len(vor.points)
    for i in range(len(vor.points)):
        p = vor.points[i]
        if img.position_is_bright(p[0],p[1]):
            flags[i] = 1
    return prune_graph(vor, flags)





"""Closest Site Graph"""
def closest_site_graph(vor : VoronoiDiagram) -> Graph:
    
    points = list()
    edges = list()
    
    index = 0
    for vid in range(len(vor.vor.vertices)):
        siteid,dist = vor.closest_site(vid)
        points.append(vor.vor.vertices[vid])
        points.append(vor.vor.points[siteid])
        edges.append([index,index+1])
        index += 2
    return Graph(points,edges)

def get_closest_dists(pids : list, vor : VoronoiDiagram) -> list:
    #result = list()
    result = [vor.closest_site(pid)[1] for pid in pids]
    '''
    for pid in pids:
        site, dist = vor.closest_site(pid)
        result.append(dist)
    '''
    return result

def get_angle(eids : list, vor : VoronoiDiagram)-> list:
    result = [vor.edge_angle(eid) for eid in eids] 
    return result
    

def get_color_list(dist : list) -> list:   
    data = np.array(dist) 
    if np.max(data) == np.min(data):
        return ["blue"]*len(dist)
    norm = (data - np.min(data)) / (np.max(data) - np.min(data))
    clist = cm.rainbow(norm)
    '''
    for c in clist:
        colors.append(cl.to_hex(c))
    '''
    return [cl.to_hex(c) for c in clist]

def get_edge_color_list(colors : list, edges:list) -> list:
    col = list()
    for e in edges:
        x = e[0]
        y = e[1]
        col1 = cl.to_rgb(colors[x])
        col2 = cl.to_rgb(colors[y])
        col.append(cl.to_hex((np.array(col1)+np.array(col2))/2))
    return col  

def get_three_color_list(dist : list) -> list:
    m = max(dist)
    norm = [1.0 if d >= m else 0.75 if d > 0 else 0 for d in dist]
    clist = cm.rainbow(norm)
    return [cl.to_hex(c) for c in clist]

def rgb_to_hex(rgb):
    return '#'+'%02x%02x%02x' % rgb


def get_random_green():
    return 51, random.randint(123,255), 51

def get_random_red():
    return random.randint(123,255), 51, 51

def get_color_from_edge(paths:list) -> list:

    dict = {}
    colors = []

    black = (0,0,0)
    gray = (128,128,128)

    for p in paths:

        if p.isNeglected:
            colors.append(gray)

        elif p.colorvalue == 10:
            colors.append(black)
        else:
            if p.clusterNumber in dict:
                colors.append(dict[p.clusterNumber])
            else:
                color = (0,0,0)
                if p.colorvalue > 0:
                    color = get_random_green()
                else:
                    color = get_random_red()
                dict[p.clusterNumber] = color
                colors.append(dict[p.clusterNumber])

    return [rgb_to_hex(c) for c in colors]

'''
colors = ["#FFFFFF","#FFFF00","#FF0000", "#000000"] 
edges = [[0,1],[0,2],[0,3],[1,2]]
print(get_edge_color_list(colors, edges))
'''
"""Binary Image Test"""
'''
rawData = np.array([
        [[0,0,0,1],[255,255,255,1],[0,0,0,1]],
    [[0,0,0,1],[255,255,255,1],[255,255,255,1]],
    [[0,0,0,1],[0,0,0,1],[0,0,0,1]]])

image = BinaryImage(rawData, 200)


vertices = get_edge_vertices(image)

vor = VoronoiDiagram(vertices)
print(vor.vor.points)
'''
'''
print(vor.region_to_site)

print(vor.vert_to_regions)
print(vor.closest_site(0))
'''
'''
vor = get_voronoi(vertices)


points = [[0,1],[1,2],[2,3],[3,4]]
edges = [[0,1],[0,2],[0,3],[1,3],[2,3],[1,2]]
flags = [0,1,1,0]

graph = Graph(points, edges)

new_graph = graph_in_image(vor, image)
print(new_graph.points)
print(new_graph.edgeIndex)
'''
'''
dist = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7]
rainbow = cm.rainbow(dist)
print(rainbow)
'''
#print(cl.to_hex(rainbow))
