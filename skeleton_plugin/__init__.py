import napari
from . import graph
from . import drawing
from . import display
from .timer import TimeRecord
from .mywidget import WidgetManager
from .pruning import ETPruningAlgo
#from magicgui import magicgui
# from napari.utils.notifications import show_info
# import testalgo as ta



def show_hello_message():
    # ta.test_boundary_edge([]);
    
    WidgetManager.inst().start()
    
    tRec = TimeRecord()
    
    tRec.stamp("Start")
    viewer = napari.current_viewer()
    layer = find_image_layer(viewer.layers)
    data = read_data(layer)
    #show_data(data)
    tRec.stamp("Read Data")
    
    biimage = graph.BinaryImage(data, 100)
    tRec.stamp("Threshold")
    
    g = graph.get_edge_vertices(biimage)
    #print(g.points)
    tRec.stamp("Find Edge")
    
    peConfig = get_vorgraph_config()
    peConfig.pointConfig.size = 0.5
    peConfig.pointConfig.edge_color = "red"
    #display.Display.current().draw_layer(graph.Graph(g,[],[]), peConfig, display.boundary)
    
    #tRec.stamp("Draw Boundary")
    
    vorGraph = graph.get_voronoi(g)
    tRec.stamp("Voronoi")
    
    peConfig = get_vorgraph_config()    
    
    #display.Display.current().draw_layer(vorGraph.graph, peConfig, display.voronoi)
    #tRec.stamp("Draw Voronoi")
    
    '''prunedGraph'''   
    prunedGraph = graph.graph_in_image(vorGraph.graph, biimage)
    #peConfig.pointConfig.name = "vor p pruned"
    #peConfig.edgeConfig.name = "vor edge pruned"
    tRec.stamp("Prune Voronoi")
    #display.Display.current().draw_layer(prunedGraph, peConfig, display.internalVoronoi)
    #tRec.stamp("Draw Prune Voronoi")
    
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
    
    #colors = graph.get_color_list(closestDist)
    #peConfig.pointConfig.edge_color = colors
    #peConfig.edgeConfig.edge_color = graph.get_edge_color_list(colors,prunedGraph.edgeIndex)
    #display.Display.current().draw_layer(prunedGraph, peConfig, display.heatmap)
    
    #tRec.stamp("Draw Heat Map")
    
    algo = ETPruningAlgo(prunedGraph, closestDist)
    algo.burn()
    
    tRec.stamp("Burn")
    
    '''
    ts = algo.npGraph.get_bts()
    colors = graph.get_color_list(bts)
    peConfig.pointConfig.edge_color = colors
    peConfig.edgeConfig.edge_color = graph.get_edge_color_list(colors,prunedGraph.edgeIndex)
    display.Display.current().draw_layer(prunedGraph, peConfig, display.burnTime)
    tRec.stamp("Draw Burn graph")
    '''   
    finalGraph = algo.prune(10) 
    tRec.stamp("ET Prune")
    
    peConfig.pointConfig.size = 2
    peConfig.pointConfig.face_color = "red"
    peConfig.pointConfig.edge_color = "red"
    peConfig.edgeConfig.size = 2
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
    
def get_vorgraph_config() -> drawing.PointEdgeConfig:
    pConfig = drawing.default_config()
    eConfig = drawing.default_config()
    
    #pConfig.name = "vor points"
    pConfig.size = 0.5
    
    #eConfig.name = "vor edges"
    eConfig.size = 0.4
    
    return drawing.PointEdgeConfig(pConfig, eConfig)



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
    