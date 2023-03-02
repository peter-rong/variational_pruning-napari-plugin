# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 13:28:18 2022

@author: Yigan
"""

from .graph import Graph, dist2D,midPoint, prune_graph, getCentroid, rgb_to_hex
from queue import PriorityQueue
import sys
from collections import deque
import time
import math
from .dynamicTree import DynamicTreeNode, DynamicTreeEdge, DynamicTree
from .dynamicTreeAlgorithm import Algorithm
from typing import Union
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


class Node:

    def __init__(self, p, r: float, ma: float):
        self.point = p
        self.radius = r
        self.bt = ma
        self.isCore = True
        self.paths = set()
        self.cost = 0
        self.posClusterIndex = -1
        self.inSol = True
        self.reward = 0

    def et(self):
        return self.bt - self.radius

    def get_one_path(self):
        for p in self.paths:
            return p
        return None

    def add_path(self, path):
        if path not in self.paths:
            self.paths.add(path)

    def remove_path(self, path):
        self.paths.remove(path)

    def is_iso(self):
        return len(self.paths) == 1

    def get_next(self, path):
        return path.other if path.one == self else path.one

    def all_path_taken(self):
        for path in self.paths:
            if not path.isTaken:
                return False
        return True

    def get_all_path_within_cluster(self, pathType) -> list:

        l = []

        if pathType == 'core+pos':
            for path in self.paths:
                if path.isTaken is False and path.segval > 0:
                    path.segval = 10
                    l.append(path)

        if pathType == 'pos':
            for path in self.paths:
                if path.isTaken is False and path.segval != 10 and path.segval > 0:
                    l.append(path)

        if pathType == 'neg':

            if self.is_junction():
                return l

            for path in self.paths:
                if path.isNeglected is False and path.isTaken is False and path.segval < 0:
                    l.append(path)

        if pathType == 'core':
            for path in self.paths:
                if path.isTaken is False and path.segval == 10:  # preset value
                    l.append(path)

        return l

    def is_explicit_junction(self):
        count = 0

        for path in self.paths:
            if path.isNeglected:
                continue
            if path.segval > 0:
                return False
            if path.segval < 0:
                count += 1
        return count > 2

    def is_junction(self):

        count = 0
        has_pos = False
        for path in self.paths:
            if path.isNeglected:
                continue
            if path.segval > 0:
                has_pos = True
            if path.segval < 0:
                count += 1
        if has_pos:
            count += 1

        return count > 2


class Path:

    def __init__(self, one: Node, other: Node, l: float):
        self.one = one
        self.other = other
        self.length = l
        self.reward = 0
        self.theta = 0
        self.segval = 0
        self.colorvalue = 0
        self.isCore = True
        self.isTaken = False
        self.isNeglected = False
        self.clusterNumber = 0
        self.clusterPointIndex = None
        self.clusterEdgeIndex = None
        self.inSol = True


class NodePathGraph:

    def to_dynamic_tree(self) -> DynamicTree:

        total_reward = 0
        min_cost = math.inf

        for node in self.nodes:
            node.isCore = False

        hasCore = False
        for path in self.paths:
            total_reward += (math.sin(path.theta)*path.length)
            min_cost = min(min_cost,path.length/2)
            if path.isCore:
                path.one.isCore = True
                path.other.isCore = True
                hasCore = True

        dynamicTree = DynamicTree([], [], [], [])

        node_map = {}

        for node in self.nodes:

            if not node.isCore:
                node_reward = 0
                node_cost = 0
                for path in node.paths:
                    node_reward += (math.sin(path.theta)*path.length /2)
                    node_cost += path.length/2

                new_node = DynamicTreeNode(node.point,node_reward,node_cost)
                dynamicTree.add_node(new_node)
                node_map[node] = new_node

        if hasCore:
            core_node = DynamicTreeNode(node.point, total_reward, min_cost)
            dynamicTree.add_node(core_node)

        for path in self.paths:
            if not path.one.isCore and not path.other.isCore:
                new_edge = DynamicTreeEdge(node_map[path.one], node_map[path.other])
                dynamicTree.add_edge(new_edge)
            elif path.one.isCore and not path.other.isCore:
                new_edge = DynamicTreeEdge(node_map[path.other],core_node)
                dynamicTree.add_edge(new_edge)
            elif path.other.isCore and not path.one.isCore:
                new_edge = DynamicTreeEdge(node_map[path.one], core_node)
                dynamicTree.add_edge(new_edge)

        return dynamicTree


    def __init__(self, points, edges, radi, ma):
        self.max = ma
        self.nodes = list()
        self.paths = list()

        for pi in range(len(points)):
            self.nodes.append(Node(p=points[pi], r=radi[pi], ma=ma))

        for e in edges:
            pid1 = e[0]
            pid2 = e[1]
            l = dist2D(points[pid1], points[pid2])
            node1 = self.nodes[pid1]
            node2 = self.nodes[pid2]
            path = Path(node1, node2, l)
            node1.add_path(path)
            node2.add_path(path)
            self.paths.append(path)

    def get_negative_degree_one_path(self) -> list():
        ans = list()

        for node in self.nodes:
            paths = list()

            for p in node.paths:
                if not p.isNeglected:
                    paths.append(p)

            if len(paths) == 1 and paths[0].segval < 0:
                ans.append(paths[0])

        return ans

    def get_degree_ones(self) -> list():
        ans = list()
        for node in self.nodes:
            if node.is_iso():
                ans.append(node)
        return ans

    def get_paths(self) -> list:
        return self.paths

    def get_ets(self) -> list:
        return [n.et() for n in self.nodes]

    def get_bts(self) -> list:
        return [n.bt for n in self.nodes]

    def get_segval(self) -> list:
        return [p.segval for p in self.paths]

    def get_colorvalue(self) -> list:
        return [p.colorvalue for p in self.paths]

    def reset_paths(self):
        for path in self.paths:
            path.one.add_path(path)
            path.other.add_path(path)

    def set_angles(self, angles: list):
        for pi in range(len(self.paths)):
            self.paths[pi].theta = angles[pi]

    def get_junction_color(self):
        return ['#0000FF' if node.is_explicit_junction() else None for node in self.nodes]


class ClusterNode:

    def __init__(self, points, r: float, core: bool, out, name: str):
        self.points = points
        self.reward = r
        self.isCore = core
        self.outerpoints = out
        self.name = name
        self.paths = list()


class ClusterPath:

    def __init__(self, one: Node, other: Node, c: float):
        self.one = one
        self.other = other
        self.cost = c


class PItem:

    def __init__(self, p: float, i):
        self.pri = p
        self.item = i

    def __lt__(self, other):
        return self.pri < other.pri


class BurningAlgo:

    def __init__(self, g: Graph, radi: list, ma: float):
        self.graph = g
        self.npGraph = NodePathGraph(g.points, g.edgeIndex, radi, ma)

    def burn(self):
        d_ones = self.npGraph.get_degree_ones()
        pq = PriorityQueue()
        for n in d_ones:
            n.bt = n.radius
            pq.put(PItem(n.bt, n))

        while not pq.empty():
            targetN = pq.get().item
            path = targetN.get_one_path()
            if path is None:
                continue
            path.isCore = False
            nextN = targetN.get_next(path)
            nextN.remove_path(path)
            if nextN.is_iso():
                nextN.bt = targetN.bt + path.length
                pq.put(PItem(nextN.bt, nextN))

        self.npGraph.reset_paths()


class PruningAlgo:

    def __init__(self, g: Graph, npg: NodePathGraph):
        self.graph = g
        self.npGraph = npg

    def prune(self, thresh: float) -> Graph:
        # virtual
        pass


class ETPruningAlgo(PruningAlgo):

    def __init__(self, g: Graph, npg: NodePathGraph):
        super().__init__(g, npg)

    def prune(self, thresh: float) -> Graph:
        # todo
        removed = set()
        d_ones = self.npGraph.get_degree_ones()
        pq = PriorityQueue()
        for n in d_ones:
            pq.put(PItem(n.et(), n))

        while not pq.empty():
            targetN = pq.get().item
            if targetN.et() >= thresh:
                break;
            removed.add(targetN)
            path = targetN.get_one_path()
            if path is None:
                continue;
            nextN = targetN.get_next(path)
            nextN.remove_path(path)
            if nextN.is_iso():
                pq.put(PItem(nextN.et(), nextN))

        self.npGraph.reset_paths()

        flags = [0 if node in removed else 1 for node in self.npGraph.nodes]
        return prune_graph(self.graph, flags)


def has_endpoint(cluster: list, node: Node):
    for path in cluster:
        if path.one == node or path.other == node:
            return True

    return False


def cluster_endpoints(c: list) -> list:
    points_list, connecting_points = list(), list()
    dict = {}
    for path in c:
        if path.one not in dict:
            dict[path.one] = 1
        else:
            dict[path.one] = dict[path.one] + 1
        if path.other not in dict:
            dict[path.other] = 1
        else:
            dict[path.other] = dict[path.other] + 1

    for key in dict.keys():
        if dict[key] == 1:
            connecting_points.append(key)

    return connecting_points


class AnglePruningAlgo(PruningAlgo):

    def __init__(self, g: Graph, npg: NodePathGraph):
        super().__init__(g, npg)

    def to_text_testing(self, graph, reward_list, cost_list: float):

        resulting_text = 'The resulting graph is:\nSECTION Graph\n'
        node_count, edge_count = len(graph.points), len(graph.edgeIndex)
        resulting_text += 'Nodes ' + str(node_count) + '\nEdges ' + str(edge_count) + '\n'

        terminal_count = 0
        terminal_text = ''

        # TODO
        # need to fix sys.maxsize/2 and convert into the total cost
        for i in range(len(graph.points)):
            if reward_list[i] != sys.maxsize / 2:
                resulting_text += 'N ' + str((i + 1)) + ' ' + str(reward_list[i]) + '\n'
            else:
                terminal_count += 1
                terminal_text += 'TP ' + str((i + 1)) + ' ' + str(0) + '\n'

        for j in range(len(graph.edgeIndex)):
            resulting_text += 'E ' + str((graph.edgeIndex[j][0] + 1)) + ' ' + str(
                (graph.edgeIndex[j][1] + 1)) + ' ' + str(abs(cost_list[j])) + '\n'

        resulting_text += 'END' + '\n' + '\n' + 'SECTION Terminals' + '\n' + 'Terminals ' + str(terminal_count) + '\n'
        resulting_text += terminal_text + 'END' + '\n' + '\n' + 'EOF'

        return resulting_text

    def generate_dynamic_tree(self):
        dynamic_tree = self.npGraph.to_dynamic_tree()
        Algorithm(dynamic_tree).execute(dynamic_tree)

    # gives output file for the dapcstp solver
    def prune(self, thresh: float):
        self.generate_dynamic_tree()
        print("There are : " + str(len(self.npGraph.nodes)) + " nodes")
        print("There are : " + str(len(self.npGraph.paths)) + " edges")
        clusters, junctions = self.__angle_thresh_cluster(thresh)
        graph, color, reward_list, cost_list, original_graph = self.generate_centroid_graph(clusters, junctions)
        print("Centroid Graph Done")

        '''
        PCST = self.to_text_testing(graph, reward_list, cost_list)
        pathname = './output/output_PCST.pcstp'
        file = open(pathname, "w")
        file.write(PCST)
        file.close()
        '''
        return graph, color, reward_list, cost_list, original_graph

    def prune_heat(self, thresh: float) -> list:
        self.__angle_thresh(thresh)
        pass

    def generate_centroid_graph(self, clusters: list, junctions: list):

        non_negative_clusters = []  # for connecting edges
        negative_clusters = []  # represesnt edge

        point_list = list()
        edge_list = list()
        point_color_list = list()
        point_reward_list = list()
        edge_cost_list = list()

        black = (0, 0, 0)
        green = (0, 255, 0)
        blue = (0, 0, 255)

        total_cost = 0

        time1 = time.time()

        for c in clusters:
            if c[0].segval < 0:
                for path in c:
                    total_cost += path.segval

        for c in clusters:
            # point_dict_key = list()
            # point_dict_value = list()
            current_mapped_point = list()

            if c[0].segval == c[0].length * 10:
                non_negative_clusters.append(c)
                curr_point = getCentroid(c)
                point_list.append(getCentroid(c))

                curr_index = len(point_list) - 1
                for path in c:
                    path.clusterPointIndex = curr_index

                # point_dict_key = tuple(getCentroid(c))

                point_color_list.append(black)
                point_reward_list.append(
                    abs(total_cost) + 1)  # abs(total_cost) +1 represents the core reward (becomes terminal)

            elif c[0].segval > 0:
                non_negative_clusters.append(c)

                curr_point = getCentroid(c)
                point_list.append(getCentroid(c))
                curr_index = len(point_list) - 1

                for path in c:
                    path.clusterPointIndex = curr_index

                # point_dict_key = tuple(getCentroid(c))

                point_color_list.append(green)
                reward = 0
                for path in c:
                    reward += path.segval
                point_reward_list.append(reward)  # reward equals sum of segval value


            else:
                negative_clusters.append(c)

        for j in junctions:
            point_list.append(j.point)
            # point_map_to_points.append(list()) #empty mapped points
            point_color_list.append(blue)  # junction points
            point_reward_list.append(0)  # 0 represents the junction point reward

        time2 = time.time()
        print(time2 - time1)

        for i in range(0, len(non_negative_clusters)):
            pos_c = non_negative_clusters[i]
            for path in pos_c:
                path.one.posClusterIndex = i
                path.other.posClusterIndex = i

        for neg_c in negative_clusters:

            cost = 0
            for path in neg_c:
                cost += path.segval

            endpoints = cluster_endpoints(neg_c)
            point_one, point_two = endpoints[0], endpoints[1]

            # pair_dict_key = []
            # pair_dict_value = list()

            edge = []

            if point_one.posClusterIndex > -1:
                clusterIndex = point_one.posClusterIndex
                edge.append(clusterIndex)
                # pair_dict_key.append(tuple(getCentroid(non_negative_clusters[clusterIndex])))

            if point_two.posClusterIndex > -1:
                clusterIndex = point_two.posClusterIndex
                edge.append(clusterIndex)
                # pair_dict_key.append(tuple(getCentroid(non_negative_clusters[clusterIndex])))

            for j in junctions:
                if j == point_one or j == point_two:
                    for i in range(0, len(point_list)):
                        if j.point[0] == point_list[i][0] and j.point[1] == point_list[i][1]:
                            edge.append(i)
                            # pair_dict_key.append(tuple(point_list[i]))
                            break
            # print(edge)

            # pair_dict_key = tuple(pair_dict_key)

            for path in neg_c:
                path.clusterEdgeIndex = len(edge_list)
                # pair_dict_value.append([path.one.point, path.other.point])

            # point_pair_map[pair_dict_key] = pair_dict_value

            edge_list.append(edge)
            edge_cost_list.append(cost)

        time3 = time.time()

        print(time3 - time2)
        return Graph(point_list, edge_list), [rgb_to_hex(c) for c in point_color_list], \
               point_reward_list, edge_cost_list, self.npGraph

    def __angle_thresh(self, thresh: float):
        pos = set()
        neg = set()
        core = set()
        for p in self.npGraph.paths:
            p.clusterEdge = None
            p.clusterPoint = None
            if p.isCore:
                p.segval = 10
                core.add(p)

            else:
                p.segval = math.sin(p.theta / 2) - math.sin(thresh / 2)
                if p.segval >= 0:
                    pos.add(p)
                else:
                    p.segval = 0
                    neg.add(p)
        return (pos, neg, core)

    def __angle_thresh_cluster(self, thresh: float):

        clusters = list()
        for node in self.npGraph.nodes:
            node.posClusterIndex = -1
            node.inSol = True

        for path in self.npGraph.paths:
            path.isTaken = False
            path.isNeglected = False
            path.inSol = True
            path.clusterNumber = 0

        for p in self.npGraph.paths:
            if p.isCore:
                p.segval = 10 * p.length  # big enough
            else:
                p.segval = (math.sin(p.theta) - math.sin(thresh)) * p.length

        path_to_neglect = self.npGraph.get_negative_degree_one_path()
        while len(path_to_neglect) > 0:
            for p in path_to_neglect:
                p.isNeglected = True
            path_to_neglect = self.npGraph.get_negative_degree_one_path()
        '''
        while len(path_to_remove) > 0:
            self.npGraph.remove_paths(path_to_remove)
            path_to_remove = self.npGraph.get_negative_degree_one_path()
        '''
        cluster_count = 0

        # get all the ones next to core
        for path in self.npGraph.paths:
            if not path.isCore:
                continue
            path_type = 'core+pos'
            cluster_count += 1
            queue = deque()
            curr_set = list()
            queue.append(path)
            path.isTaken = True
            path.clusterNumber = cluster_count

            while len(queue) > 0:
                curr_path = queue.popleft()
                curr_set.append(curr_path)

                first_node = curr_path.one
                second_node = curr_path.other

                l1 = first_node.get_all_path_within_cluster(path_type)
                l2 = second_node.get_all_path_within_cluster(path_type)

                for path in l1:
                    queue.append(path)
                    path.isTaken = True
                    path.clusterNumber = cluster_count

                for path in l2:
                    queue.append(path)
                    path.isTaken = True
                    path.clusterNumber = cluster_count

            color_total_value = 0
            for path in curr_set:
                color_total_value += path.segval
            for path in curr_set:
                path.colorvalue = color_total_value / len(curr_set)
            clusters.append(curr_set)
            break

        for path in self.npGraph.paths:
            if path.isNeglected:
                continue
            if path.isTaken:
                continue
            cluster_count += 1
            path_type = ''
            queue = deque()
            curr_set = list()
            if path.isCore:
                print('This path is a core and this should not ever print')
                path_type = 'core'
            elif path.segval > 0:
                path_type = 'pos'
            else:
                path_type = 'neg'
            queue.append(path)
            path.isTaken = True
            path.clusterNumber = cluster_count

            while len(queue) > 0:
                curr_path = queue.popleft()
                curr_set.append(curr_path)

                first_node = curr_path.one
                second_node = curr_path.other

                l1 = first_node.get_all_path_within_cluster(path_type)
                l2 = second_node.get_all_path_within_cluster(path_type)

                for path in l1:
                    queue.append(path)
                    path.isTaken = True
                    path.clusterNumber = cluster_count

                for path in l2:
                    queue.append(path)
                    path.isTaken = True
                    path.clusterNumber = cluster_count

            color_total_value = 0
            for path in curr_set:
                color_total_value += path.segval / path.length

            for path in curr_set:
                path.colorvalue = color_total_value / len(curr_set)

            clusters.append(curr_set)

        self.npGraph.nonTakenPaths = self.npGraph.paths
        for path in self.npGraph.paths:
            path.isTaken = False

        print('There are ' + str(len(clusters)) + ' clusters')

        junctions = list()

        for node in self.npGraph.nodes:
            if node.is_explicit_junction():
                junctions.append(node)

        print('There are ' + str(len(junctions)) + ' junctions')

        return clusters, junctions

    # return pos_cluster, neg_cluster, core_cluster
    # not used at this point
    def generate_graph(self, clusters, junctions):

        cluster_nodes = list()
        cluster_paths = list()
        junction_counter = 0
        pos_counter = 0

        # junction node
        for j in junctions:
            junction_counter += 1
            points_list, outerpoints_list = list(j.point), list(j.point)
            point_name = 'junction ' + str(junction_counter)
            cluster_node = ClusterNode(points_list, 0, False, outerpoints_list, point_name)
            cluster_nodes.append(cluster_node)

        for c in clusters:
            # positive or core node
            if c[0].colorvalue / c[0].length > 0:
                is_core = False
                point_name = ''
                if c[0].colorvalue / c[0].length == 10:
                    is_core = True
                    point_name = 'core'
                else:
                    pos_counter += 1
                    point_name = 'pos ' + str(pos_counter)

                points_list, outerpoints_list = list(), list()
                dict = {}
                for path in c:
                    if path.one not in points_list:
                        points_list.append(path.one)
                    if path.other not in points_list:
                        points_list.append(path.other)
                    if path.one not in dict:
                        dict[path.one] = 1
                    else:
                        dict[path.one] = dict[path.one] + 1
                    if path.other not in dict:
                        dict[path.other] = 1
                    else:
                        dict[path.other] = dict[path.other] + 1

                for key in dict.keys():
                    if dict[key] == 1:
                        outerpoints_list.append(key)

                cluster_node = ClusterNode(points_list, len(c) * c[0].colorvalue, is_core, outerpoints_list, point_name)
                cluster_nodes.append(cluster_node)

            # negative as path
            else:
                outerpoints_list = []
                dict = {}
                for path in c:
                    if path.one not in dict:
                        dict[path.one] = 1
                    else:
                        dict[path.one] = dict[path.one] + 1
                    if path.other not in dict:
                        dict[path.other] = 1
                    else:
                        dict[path.other] = dict[path.other] + 1
                for key in dict.keys():
                    if dict[key] == 1:
                        outerpoints_list.append(key)

                cluster_path = ClusterPath(outerpoints_list[0], outerpoints_list[1], len(c) * c[0].colorvalue)
                outerpoints_list[0].paths.append(cluster_path)  # set path
                outerpoints_list[1].paths.append(cluster_path)  # set path
                cluster_paths.append(cluster_path)

        # graph with networkx


