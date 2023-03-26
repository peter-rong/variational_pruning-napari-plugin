# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 15:35:25 2022

@author: Yigan
"""
import math

from . import graph
from . import statemachine as st
from . import mainalgo as ma
from . import display as ds
from .pruning import BurningAlgo, ETPruningAlgo, AnglePruningAlgo
from . import tree
from . import treealgorithm
import numpy as np


def algo_st():
    return ma.SkeletonApp.inst().algoStatus


def app_st():
    return ma.SkeletonApp.inst().appStatus


def tRec():
    return ma.SkeletonApp.inst().timer


def get_size() -> float:
    refer = 128
    x, y, c = app_st().shape
    m = float(max([x, y, c]))
    return m / refer


class ReadState(st.State):

    def execute(self):
        algo_st().raw_data = self.__read_data()
        app_st().shape = algo_st().raw_data.shape
        tRec().stamp("Read Data")

    def get_next(self):
        return ThreshState()

    def __read_data(self):
        viewer = ds.Display.current().viewer
        layer = viewer.layers[0]
        return layer.data_raw


class ThreshState(st.State):

    def execute(self):
        if algo_st().raw_data is None:
            return
        algo_st().biimg = graph.BinaryImage(algo_st().raw_data, int(app_st().biThresh / 100.0 * 255))
        tRec().stamp("Threshold")

    def get_next(self):
        if algo_st().raw_data is None:
            return None
        return BoundaryState()


class BoundaryState(st.State):

    def execute(self):
        algo_st().boundary = graph.get_edge_vertices(algo_st().biimg)
        tRec().stamp("Find Edge")

        peConfig = ma.get_vorgraph_config(get_size())
        peConfig.pointConfig.edge_color = "red"
        ds.Display.current().draw_layer(graph.Graph(algo_st().boundary, [], []), peConfig, ds.boundary)
        tRec().stamp("Draw Boundary")

    def get_next(self):
        return VorState()


class VorState(st.State):

    def execute(self):
        algo_st().vor = graph.get_voronoi(algo_st().boundary)
        tRec().stamp("Voronoi")

    def get_next(self):
        return PruneState()


class PruneState(st.State):

    def execute(self):
        algo_st().graph = graph.graph_in_image(algo_st().vor.graph, algo_st().biimg)
        tRec().stamp("Prune Voronoi")

        peConfig = ma.get_vorgraph_config(get_size())
        ds.Display.current().draw_layer(algo_st().vor.graph, peConfig, ds.internalVoronoi)
        tRec().stamp("Draw Prune Voronoi")

    def get_next(self):
        return BTState()


class BTState(st.State):

    def execute(self):
        closestDist = graph.get_closest_dists(algo_st().graph.point_ids, algo_st().vor)
        tRec().stamp("Calc Radius")

        algo_st().algo = BurningAlgo(algo_st().graph, closestDist, max(app_st().shape))
        algo_st().algo.burn()
        tRec().stamp("Burn")

        bts = algo_st().algo.npGraph.get_bts()
        ets = algo_st().algo.npGraph.get_ets()
        self.__draw(bts, ds.burnTime)
        self.__draw(ets, ds.erosionT)
        tRec().stamp("Draw Burn Graph")

    def get_next(self):
        return PruneChoosingState()

    def __draw(self, radi, layerName):
        peConfig = ma.get_vorgraph_config(get_size())
        colors = graph.get_color_list(radi)
        peConfig.pointConfig.edge_color = colors
        peConfig.edgeConfig.edge_color = graph.get_edge_color_list(colors, algo_st().graph.edgeIndex)
        ds.Display.current().draw_layer(algo_st().graph, peConfig, layerName)


class PruneChoosingState(st.State):

    def get_next(self):
        return ETPruneState() if app_st().method == 0 else AngleState()


class AngleState(st.State):
    def execute(self):
        if algo_st().algo is None:
            return

        angles = graph.get_angle(algo_st().graph.edge_ids, algo_st().vor)
        # print(angles)
        algo_st().algo.npGraph.set_angles(angles)

        tRec().stamp("calc angles")

    def get_next(self):
        return AnglePruneState()


class ETPruneState(st.State):

    def execute(self):
        if algo_st().algo is None:
            return

        # outputing skeleton graph as a txt file
        if ds.Display.current().config.flag_raise(ds.outputSkeleton):
            algo_st().algo.npGraph.to_skeleton_text_file()
            print("I'm here")

        prune_algo = ETPruningAlgo(algo_st().algo.graph, algo_st().algo.npGraph)
        pruneT = app_st().etThresh / 100.0 * max(app_st().shape)
        algo_st().final = prune_algo.prune(pruneT)
        tRec().stamp("ET Prune")

        peConfig = ma.get_vorgraph_config(get_size())
        peConfig.pointConfig.face_color = "red"
        peConfig.pointConfig.edge_color = "red"
        peConfig.edgeConfig.face_color = "red"
        peConfig.edgeConfig.edge_color = "red"

        ds.Display.current().draw_layer(algo_st().final, peConfig, ds.final)
        tRec().stamp("Draw Final")


class AnglePruneState(st.State):

    def execute(self):
        if algo_st().algo is None:
            return

        pruneT = np.pi * app_st().etThresh / 100.0
        raw_threshold = app_st().etThresh / 100.0

        angle_graph, original_reward_list = algo_st().algo.npGraph.to_graph_angle()

        print("threshold is " + str(raw_threshold))

        prune_algo = AnglePruningAlgo(algo_st().algo.graph, algo_st().algo.npGraph)
        tRec().stamp("start of angle function and cluster")
        centroid_graph, centroid_points_color, reward_list, cost_list, original_graph = prune_algo.prune(pruneT)
        tRec().stamp("compute angle function and cluster")

        dynamic_tree, dynamic_tree_list, dynamic_reward_list, dynamic_alpha_list = prune_algo.dynamic_prune(
            raw_threshold)
        dynamic_graph = dynamic_tree.to_graph()

        '''
        dynamic_full_graph, dynamic_color_list = dynamic_tree.to_colored_graph(dynamic_tree_list)
        dynamic_color_list = graph.get_color_list(dynamic_color_list)
        
        
        dynamicConfig = ma.get_dynamic_result_config(get_size())
        dynamicFullConfig = ma.get_dynamic_result_config(get_size())

        dynamicFullConfig.edgeConfig.edge_color = dynamic_color_list

        tRec().stamp("compute dynamic tree list")

        ds.Display.current().draw_layer(dynamic_graph, dynamicConfig, ds.dynamic)

        
        dynamicConfig.pointConfig.face_color = graph.get_color_list(dynamic_reward_list)  # testing
        dynamicConfig.pointConfig.edge_color = graph.get_color_list(dynamic_reward_list)  # testing
        ds.Display.current().draw_layer(dynamic_full_graph, dynamicConfig, ds.dynamic)  # testing
        
        tRec().stamp("draw dynamic graph")

        ds.Display.current().draw_layer(dynamic_full_graph, dynamicFullConfig, ds.full_dynamic)
        tRec().stamp("draw full dynamic graph")
        '''

        peConfig = ma.get_angular_config(get_size())
        centroid_peConfig = ma.get_angular_centroid_config(get_size())

        all_edge = algo_st().algo.npGraph.get_paths()
        edge_colors = graph.get_color_from_edge(all_edge)

        point_colors = algo_st().algo.npGraph.get_junction_color()

        peConfig.pointConfig.edge_color = point_colors
        peConfig.pointConfig.face_color = point_colors
        peConfig.edgeConfig.edge_color = edge_colors

        peTestConfig = ma.get_angular_config(get_size())
        peTestConfig.edgeConfig.edge_color = graph.get_color_list(original_reward_list)
        peTestConfig.pointConfig.edge_color = 'blue'
        peTestConfig.pointConfig.face_color = 'blue'

        ds.Display.current().draw_layer(angle_graph, peTestConfig, ds.angle)  # test
        # ds.Display.current().draw_layer(algo_st().graph, peConfig, ds.angle) TODO to turn back
        tRec().stamp("draw cluster by angles")

        centroid_peConfig.pointConfig.edge_color = centroid_points_color
        centroid_peConfig.pointConfig.face_color = centroid_points_color
        centroid_peConfig.edgeConfig.edge_color = "purple"

        tRec().stamp("before_draw_PCST")
        ds.Display.current().draw_layer(centroid_graph, centroid_peConfig, ds.pcst)
        tRec().stamp("draw_initial_PCST")

        initial_tree = tree.Tree(centroid_graph.points, centroid_graph.edgeIndex, reward_list, cost_list)
        result_tree = treealgorithm.Algorithm(initial_tree).execute()

        PCST_result_graph = result_tree.to_graph(centroid_graph)
        tRec().stamp("PCST_solver")
        PCST_result_peConfig = ma.get_PCST_result_config(get_size())

        skeleton_result_graph = graph.cluster_to_skeleton(PCST_result_graph, original_graph)
        tRec().stamp("PCST_to_Graph")

        ds.Display.current().draw_layer(PCST_result_graph, PCST_result_peConfig, ds.pcstResult)
        tRec().stamp("draw_PCST_result")

        # TODO
        # skeleton_result_peConfig = ma.get_skeleton_result_config(get_size())
        # ds.Display.current().draw_layer(skeleton_result_graph, skeleton_result_peConfig, ds.skeletonResult)
        # tRec().stamp("draw_skeleton_result")


class ResponseState(st.State):
#TODO
    def execute(self):
        return
