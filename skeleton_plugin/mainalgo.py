# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 15:11:46 2022

@author: Yigan
"""

import napari
from . import graph
from . import drawing
from . import display
from .timer import TimeRecord
from .pruning import ETPruningAlgo
from .statemachine import StateMachine
from . import appstates as aps


class AlgoStatus:

    def __init__(self):
        self.curveGraph = None
        self.raw_data = None
        self.biimg = None
        self.boundary = None
        self.vor = None
        self.graph = None
        self.algo = None
        self.final = None


class AppStatus:

    def __init__(self):
        self.biThresh = 0
        self.etThresh = 0
        self.vaThresh = 0
        self.shape = None


class SkeletonApp:
    drawMode = None

    __current = None
    hasSolution = False
    graph = None
    ets = None
    dynamic_graph = None
    threshold_list = None
    color_list = None

    output = None

    def __init__(self):
        self.algoStatus = AlgoStatus()
        self.appStatus = AppStatus()
        self.stm = StateMachine()
        self.timer = TimeRecord()
        pass

    def inst():
        if SkeletonApp.__current is None:
            SkeletonApp.__current = SkeletonApp()
        return SkeletonApp.__current

    def run(self):
        self.timer.clear()
        self.timer.stamp("Start")
        self.hasSolution = False
        self.drawMode = None
        self.graph = None
        self.ets = None
        self.dynamic_graph = None
        self.threshold_list = None
        self.color_list = None
        self.output = None

        self.stm.change_state(aps.ReadState())

        self.__runall()
        self.hasSolution = True

        self.timer.print_records()

    def file_to_graph(self, filename, fairing_count):

        file = open(filename, 'r')
        lines = file.readlines()
        node_count = int(lines[0])
        points = []

        for i in range(1, node_count + 1):
            line = lines[i]
            p1, p2 = float(line.split()[0]), float(line.split()[1])
            points.append([p1, p2])

        edge_ids = []

        for j in range(node_count + 2, len(lines)):
            line = lines[j]
            id1, id2 = int(line.split()[0]), int(line.split()[1])
            edge_ids.append([id1, id2])

        #fairing
        while fairing_count > 0:
            new_points = list()

            for i in range(0,len(points)):
                new_points.append([0,0])

            for edge in edge_ids:
                p1, p2 = edge[0], edge[1]

                new_points[p1][0] += (points[p2][0]/2.0)
                new_points[p1][1] += (points[p2][1]/2.0)

                new_points[p2][0] += (points[p1][0]/2.0)
                new_points[p2][1] += (points[p1][1]/2.0)

            points = new_points
            fairing_count -= 1

        curve_graph = graph.Graph(points, edge_ids)

        return curve_graph

    def load_curve(self, filename, fairing_count):

        curve_graph = self.file_to_graph(filename, fairing_count)

        peConfig = get_vorgraph_config(0.5)
        peConfig.pointConfig.edge_color = "red"

        self.algoStatus.curveGraph = curve_graph

        display.Display.current().draw_edge_layer(curve_graph, peConfig, "curve")

        self.stm.change_state(aps.CurveVorState())

        while self.stm.valid():
            self.stm.execute()
            self.stm.to_next()
        self.timer.print_records()

        '''
        self.timer.clear()
        self.timer.stamp("Start loading curve")
        self.hasSolution = False

        self.stm.change_state(aps.ReadCurveState())
        self.__runall()
        self.hasSolution = True

        self.timer.print_records()
        '''

    def va_color(self):
        if not self.hasSolution:
            print("No solution yet, cannot draw graph")

        else:
            SkeletonApp.drawMode = 'va_color'

            self.stm.change_state(aps.VaColorState())
            self.stm.execute()
            self.timer.print_records()

    def va_prune(self):
        if not self.hasSolution:
            print("No solution yet, cannot draw graph")

        else:
            SkeletonApp.drawMode = 'va_prune'

            self.stm.change_state(aps.VaPruneState())
            self.stm.execute()
            self.timer.print_records()

    def et_color(self):
        if not self.hasSolution:
            print("No solution yet, cannot draw graph")

        else:

            SkeletonApp.drawMode = 'et_color'

            self.stm.change_state(aps.EtColorState())
            self.stm.execute()
            self.timer.print_records()

    def et_prune(self):
        if not self.hasSolution:
            print("No solution yet, cannot draw graph")

        else:

            SkeletonApp.drawMode = 'et_prune'

            self.stm.change_state(aps.EtPruneState())
            self.stm.execute()
            self.timer.print_records()

    def va_export(self):

        if not self.hasSolution:
            print("No solution yet, cannot export")
            return
        if SkeletonApp.drawMode == 'et_prune' or SkeletonApp.drawMode == 'et_color':
            print("You are trying to export for VA while the current solution is for ET")
            return

        if self.output is not None:
            f = open("va_output.txt", "w")
            f.write(self.output)

            f.close()

    def et_export(self):

        if not self.hasSolution:
            print("No solution yet, cannot export")
            return
        if SkeletonApp.drawMode == 'va_prune' or SkeletonApp.drawMode == 'va_color':
            print("You are trying to export for ET while the current solution is for VA")
            return

        if self.output is not None:
            f = open("et_output.txt", "w")
            f.write(self.output)

            f.close()

    def reset_vathresh(self, newT: float):
        self.appStatus.vaThresh = newT

        if SkeletonApp.drawMode == 'va_color':
            self.va_color()

        elif SkeletonApp.drawMode == 'va_prune':
            self.va_prune()

    def reset_etthresh(self, newT: float):
        self.appStatus.etThresh = newT

        if SkeletonApp.drawMode == 'et_color':
            self.et_color()

        elif SkeletonApp.drawMode == 'et_prune':
            self.et_prune()

    def reset_bithresh(self, newT: float):
        self.appStatus.biThresh = newT

        self.run()

    def __runall(self):
        while self.stm.valid():
            self.stm.execute()
            self.stm.to_next()


def run():
    # ta.test_boundary_edge([]);

    tRec = TimeRecord()

    tRec.stamp("Start")
    viewer = napari.current_viewer()
    layer = find_image_layer(viewer.layers)
    data = read_data(layer)
    # show_data(data)
    tRec.stamp("Read Data")

    size = getSize(data.shape)

    biimage = graph.BinaryImage(data, 100)
    tRec.stamp("Threshold")

    g = graph.get_edge_vertices(biimage)
    # print(g.points)
    tRec.stamp("Find Edge")

    peConfig = get_vorgraph_config(size)
    peConfig.pointConfig.edge_color = "red"
    display.Display.current().draw_layer(graph.Graph(g, [], []), peConfig, display.boundary)

    tRec.stamp("Draw Boundary")

    vorGraph = graph.get_voronoi(g)
    tRec.stamp("Voronoi")

    display.Display.current().draw_layer(vorGraph.graph, peConfig, display.voronoi)
    tRec.stamp("Draw Voronoi")

    '''prunedGraph'''
    prunedGraph = graph.graph_in_image(vorGraph.graph, biimage)
    peConfig.pointConfig.name = "vor p pruned"
    peConfig.edgeConfig.name = "vor edge pruned"
    tRec.stamp("Prune Voronoi")
    display.Display.current().draw_layer(prunedGraph, peConfig, display.internalVoronoi)
    tRec.stamp("Draw Prune Voronoi")

    '''closest site'''
    '''
    csiteGraph = graph.closest_site_graph(vorGraph)
    peConfig.pointConfig.name = "closest site p"
    peConfig.edgeConfig.name = "cloest sie e"
    peConfig.pointConfig.edge_color = "yellow"
    peConfig.edgeConfig.edge_color = "yellow"
    draw_graph(viewer, csiteGraph, peConfig)
    '''
    closestDist = graph.get_closest_dists(prunedGraph.point_ids, vorGraph)
    tRec.stamp("Calc Heat Map")

    colors = graph.get_color_list(closestDist)
    peConfig.pointConfig.edge_color = colors
    peConfig.edgeConfig.edge_color = graph.get_edge_color_list(colors, prunedGraph.edgeIndex)
    display.Display.current().draw_layer(prunedGraph, peConfig, display.heatmap)

    tRec.stamp("Draw Heat Map")

    algo = ETPruningAlgo(prunedGraph, closestDist)
    algo.burn()

    tRec.stamp("Burn")

    bts = algo.npGraph.get_bts()
    colors = graph.get_color_list(bts)
    peConfig.pointConfig.edge_color = colors
    peConfig.edgeConfig.edge_color = graph.get_edge_color_list(colors, prunedGraph.edgeIndex)
    display.Display.current().draw_layer(prunedGraph, peConfig, display.burnTime)
    tRec.stamp("Draw Burn graph")

    finalGraph = algo.prune(getThresh())
    tRec.stamp("ET Prune")

    ets = algo.npGraph.get_ets()
    colors = graph.get_color_list(ets)
    peConfig.pointConfig.edge_color = colors
    peConfig.edgeConfig.edge_color = graph.get_edge_color_list(colors, prunedGraph.edgeIndex)
    display.Display.current().draw_layer(prunedGraph, peConfig, display.erosionT)

    peConfig.pointConfig.face_color = "red"
    peConfig.pointConfig.edge_color = "red"
    peConfig.edgeConfig.face_color = "red"
    peConfig.edgeConfig.edge_color = "red"
    display.Display.current().draw_layer(finalGraph, peConfig, display.final)
    tRec.stamp("Draw Final")

    display.Display.current().reset()

    tRec.print_records()


# TODO
# step 1 ：Find image layer
# step 2 : Read raw data
# step 3 : Do something

def find_image_layer(layers):
    return layers[0]


def read_data(layer):
    return layer.data_raw
    # pass


def show_data(data):
    napari.utils.notifications.show_info(str(type(data)));
    print(data)


def get_vorgraph_config(size: float) -> drawing.PointEdgeConfig:
    pConfig = drawing.default_config()
    eConfig = drawing.default_config()

    # pConfig.name = "vor points"
    pConfig.size = size

    # eConfig.name = "vor edges"
    eConfig.size = size

    return drawing.PointEdgeConfig(pConfig, eConfig)


def get_thickness_config(size: float) -> drawing.PointEdgeConfig:
    pConfig = drawing.default_config()
    eConfig = drawing.default_config()

    pConfig.size = size

    eConfig.size = size

    peConfig = drawing.PointEdgeConfig(pConfig, eConfig)

    peConfig.pointConfig.face_color = "red"
    peConfig.pointConfig.edge_color = "red"
    peConfig.edgeConfig.face_color = "red"
    peConfig.edgeConfig.edge_color = "red"

    return peConfig


def get_angle_config(size: float) -> drawing.PointEdgeConfig:
    pConfig = drawing.default_config()
    eConfig = drawing.default_config()

    pConfig.size = size

    eConfig.size = size

    peConfig = drawing.PointEdgeConfig(pConfig, eConfig)

    peConfig.pointConfig.face_color = "orange"
    peConfig.pointConfig.edge_color = "orange"
    peConfig.edgeConfig.face_color = "orange"
    peConfig.edgeConfig.edge_color = "orange"

    return peConfig


def get_erosion_config(size: float) -> drawing.PointEdgeConfig:
    pConfig = drawing.default_config()
    eConfig = drawing.default_config()

    pConfig.size = size

    eConfig.size = size

    peConfig = drawing.PointEdgeConfig(pConfig, eConfig)

    peConfig.pointConfig.face_color = "blue"
    peConfig.pointConfig.edge_color = "blue"
    peConfig.edgeConfig.face_color = "blue"
    peConfig.edgeConfig.edge_color = "blue"

    return peConfig


def get_dynamicGraph_config(size: float) -> drawing.PointEdgeConfig:
    pConfig = drawing.default_config()
    eConfig = drawing.default_config()

    pConfig.size = size

    eConfig.size = size

    peConfig = drawing.PointEdgeConfig(pConfig, eConfig)

    peConfig.pointConfig.face_color = "green"
    peConfig.pointConfig.edge_color = "green"
    peConfig.edgeConfig.face_color = "green"
    peConfig.edgeConfig.edge_color = "green"

    return peConfig


def get_dynamic_config(size: float) -> drawing.PointEdgeConfig:
    pConfig = drawing.default_config()
    eConfig = drawing.default_config()

    pConfig.size = size

    eConfig.size = size

    peConfig = drawing.PointEdgeConfig(pConfig, eConfig)

    peConfig.pointConfig.face_color = "green"
    peConfig.pointConfig.edge_color = "green"
    peConfig.edgeConfig.face_color = "green"
    peConfig.edgeConfig.edge_color = "green"

    return peConfig


def get_angular_centroid_config(size: float) -> drawing.PointEdgeConfig:
    pConfig = drawing.angular_default_config()
    eConfig = drawing.angular_default_config()

    # pConfig.name = "vor points"
    pConfig.size = size * 3

    # eConfig.name = "vor edges"
    eConfig.size = size

    return drawing.PointEdgeConfig(pConfig, eConfig)


def get_PCST_result_config(size: float) -> drawing.PointEdgeConfig:
    pConfig = drawing.angular_default_config()
    eConfig = drawing.angular_default_config()

    # pConfig.name = "vor points"
    pConfig.size = size * 2
    pConfig.edge_color = 'red'
    pConfig.face_color = 'red'

    # eConfig.name = "vor edges"
    eConfig.size = size
    eConfig.edge_color = 'red'
    eConfig.face_color = 'red'

    return drawing.PointEdgeConfig(pConfig, eConfig)


def get_skeleton_result_config(size: float) -> drawing.PointEdgeConfig:
    pConfig = drawing.angular_default_config()
    eConfig = drawing.angular_default_config()

    # pConfig.name = "vor points"
    pConfig.size = size
    pConfig.edge_color = 'purple'
    pConfig.face_color = 'purple'

    # eConfig.name = "vor edges"
    eConfig.size = size / 2
    eConfig.edge_color = 'purple'
    eConfig.face_color = 'purple'

    return drawing.PointEdgeConfig(pConfig, eConfig)


def get_dynamic_result_config(size: float) -> drawing.PointEdgeConfig:
    pConfig = drawing.angular_default_config()
    eConfig = drawing.angular_default_config()

    pConfig.size = size
    pConfig.edge_color = 'red'
    pConfig.face_color = 'red'

    eConfig.size = size / 2
    eConfig.edge_color = 'red'
    eConfig.face_color = 'red'

    return drawing.PointEdgeConfig(pConfig, eConfig)


def get_angular_config(size: float) -> drawing.PointEdgeConfig:
    pConfig = drawing.angular_default_config()
    eConfig = drawing.angular_default_config()

    # pConfig.name = "vor points"
    pConfig.size = size * 3

    # eConfig.name = "vor edges"
    eConfig.size = size

    return drawing.PointEdgeConfig(pConfig, eConfig)


def getSize(shape) -> float:
    refer = 128
    x, y, c = shape
    m = float(max([x, y, c]))
    return m / refer


def getThresh() -> float:
    return SkeletonApp.etThresh


'''
def draw_graph(viewer : napari.Viewer, g : graph.Graph, config : drawing.PointEdgeConfig) :
    pc = config.pointConfig
    viewer.add_points(g.points, name = pc.name, size = pc.size, opacity = pc.opacity, face_color = pc.face_color, edge_color = pc.edge_color)
    
    ec = config.edgeConfig
    #viewer.add_shapes(g.get_edge_cord(), name = ec.name, scale = ec.size, opacity = ec.opacity, face_color = ec.face_color, edge_color = ec.edge_color, shape_type = "line")
    shapeLayer = napari.layers.Shapes(name = ec.name)
    shapeLayer.add_lines(g.get_edge_cord(), edge_width = ec.size, face_color = ec.face_color, edge_color = ec.edge_color)
    
    circs = list()
    for p in g.points:
        circs.append([list(p),[pc.size,pc.size]])
    shapeLayer.add_ellipses(circs, face_color = pc.face_color, edge_color = pc.edge_color)
    
    viewer.add_layer(shapeLayer)
    '''
'''
def create_widget():
    return MyWidget(napari.current_viewer())

class MyWidget(magicgui.widgets.Widget):
    """Any QtWidgets.QWidget or magicgui.widgets.Widget subclass can be used."""

    def __init__(self, viewer: "napari.viewer.Viewer", parent=None):
        super().__init__(parent)  
'''
