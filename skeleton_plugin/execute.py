import sys

import imageio

from skeleton_plugin import graph
from skeleton_plugin.graph import get_closest_dists, get_angle, prune_graph_from_edge
from skeleton_plugin.mainalgo import file_to_graph
from skeleton_plugin.pruning import BurningAlgo, AnglePruningAlgo

if len(sys.argv) != 8:
    sys.exit("Incorrect command format, sample command: python3 execute.py image toy.png 0 VA 45 prune output.txt")

file_type, filename = sys.argv[1], sys.argv[2]
file = open(filename, 'r')
binary_threshold, fairing_count, pruning_method, pruning_threshold, output_type, output_filename = \
    None, None, None, None, None, None

if file_type == "image":
    if 0 <= int(sys.argv[3]) <= 255:
        binary_threshold = int(sys.argv[3])
    else:
        sys.exit("Invalid binary threshold, valid threshold is between 0 and 255")
elif file_type == "curve":
    if int(sys.argv[3]) >= 0:
        fairing_count = int(sys.argv[3])
    else:
        sys.exit("Invalid fairing number, valid fairing number is a non-negative integer")
else:
    sys.exit("Invalid file type, valid types are image and curve")

if sys.argv[4] == "VA" or sys.argv[4] == "ET":
    pruning_method = sys.argv[4]
else:
    sys.exit("Invalid pruning method, valid methods are VA and ET")

if pruning_method == "ET" and 0 <= float(sys.argv[5]) <= 1:
    pruning_threshold = float(sys.argv[5])
elif pruning_method == "VA" and 0 <= float(sys.argv[5]) < 90:
    pruning_threshold = float(sys.argv[5])
else:
    sys.exit("Invalid pruning threshold, valid threshold for VA is between 0 and 90, for ET is between 0 and 1")

if sys.argv[6] == "color" or sys.argv[6] == "prune":
    output_type = sys.argv[6]
else:
    sys.exit("Invalid output type, valid types are color and prune")

output_filename = sys.argv[7]

# ReadState to VorState
if file_type == "image":
    image_raw_data = imageio.imread(filename)
    shape = image_raw_data.shape
    binaryImg = graph.BinaryImage(image_raw_data, binary_threshold)
    boundary = graph.get_edge_vertices(binaryImg)
    vor = graph.get_voronoi(boundary)
    graph = graph.graph_in_image(vor.graph, binaryImg)

else:
    curve_graph = file_to_graph(filename, fairing_count)

    for p in curve_graph.points:
        max_x = max(p[0], max_x)
        min_x = min(p[0], min_x)
        max_y = max(p[1], max_y)
        min_y = min(p[1], min_y)

    shape = (max_x - min_x, max_y - min_y, 3)
    vor = graph.get_voronoi(curve_graph.points)
    graph = graph.graph_in_curve(vor.graph, curve_graph)

#BTState and Execute State
closestDist = get_closest_dists(graph.point_ids, vor)
algo = BurningAlgo(graph, closestDist, max(shape))
algo.burn()
ets = algo.npGraph.get_ets()

dynamic_graph = algo.graph.duplicate()
dynamic_npGraph = algo.npGraph.duplicate()

angles = get_angle(graph.edge_ids, vor)

dynamic_npGraph.set_angles(angles)
dynamic_prune_algo = AnglePruningAlgo(dynamic_graph, dynamic_npGraph)

dynamic_tree, dynamic_reward_list, dynamic_alpha_list = dynamic_prune_algo.dynamic_prune(0)

dynamic_graph_with_threshold, dynamic_graph_thresh, dynamic_color_list_with_threshold = \
    dynamic_prune_algo.dynamic_to_graph_with_color()

#ET and VA Color/Prune States
if pruning_method == "ET" and output_type == "color":
    # get result
    resulting_text = ''
    node_count, edge_count = len(graph.points), len(graph.edgeIndex)
    resulting_text += str(node_count) + '\n'

    for i in range(len(graph.points)):
        resulting_text += str(graph.points[i][0]) + ' ' \
                          + str(graph.points[i][1]) + ' ' + str(ets[i]) + '\n'

    resulting_text += str(edge_count) + '\n'

    for edge in graph.edgeIndex:
        resulting_text += str(edge[0]) + ' ' + str(edge[1]) + '\n'

elif pruning_method == "ET" and output_type == "prune":
    curr_max = 0
    for e in graph.edgeIndex:
        x = e[0]
        y = e[1]
        curr_max = max(curr_max, (ets[x] + ets[y]) / 2)

    et_threshold = curr_max * pruning_threshold

    flags = [0 if ets[i] < et_threshold else 1 for i in range(len(graph.points))]
    Et_result_graph = graph.prune_graph(graph, flags)

    point_in_solution = [0] * len(graph.points)

    for i in range(len(ets)):
        if ets[i] >= et_threshold:
            point_in_solution[i] = 1

    node_count = sum(point_in_solution)
    edge_count = 0

    for edge in graph.edgeIndex:
        if point_in_solution[edge[0]] == 1 and point_in_solution[edge[1]] == 1:
            edge_count += 1

    # get result
    resulting_text = ''
    resulting_text += str(node_count) + '\n'

    for i in range(len(graph.points)):
        if point_in_solution[i] == 1:
            resulting_text += str(graph.points[i][0]) + ' ' \
                              + str(graph.points[i][1]) + ' ' + str(ets[i]) + '\n'

    resulting_text += str(edge_count) + '\n'

    for edge in graph.edgeIndex:
        if point_in_solution[edge[0]] == 1 and point_in_solution[edge[1]] == 1:
            resulting_text += str(edge[0]) + ' ' + str(edge[1]) + '\n'

elif pruning_method == "VA" and output_type == "color":
    dynamic_threshold = pruning_threshold / 90.0

    # get result
    resulting_text = ''
    node_count, edge_count = len(dynamic_graph.points), len(dynamic_graph.edgeIndex)
    resulting_text += str(node_count) + '\n'

    for i in range(len(dynamic_graph.points)):
        resulting_text += str(dynamic_graph.points[i][0]) + ' ' \
                          + str(dynamic_graph.points[i][1]) + '\n'

    resulting_text += str(edge_count) + '\n'

    for i in range(len(dynamic_graph.edgeIndex)):
        edge = dynamic_graph.edgeIndex[i]
        resulting_text += str(edge[0]) + ' ' + str(edge[1]) \
                          + ' ' + str(dynamic_graph_thresh[i]) + '\n'

else:
    dynamic_threshold = pruning_threshold / 90.0

    dynamic_result_graph = prune_graph_from_edge(dynamic_graph, dynamic_graph_thresh, dynamic_threshold)

    point_in_solution = [0] * len(dynamic_graph.points)
    edge_count = 0

    for i in range(len(dynamic_graph_thresh)):
        if dynamic_graph_thresh[i] >= dynamic_threshold:
            index1, index2 = dynamic_graph.edgeIndex[i][0], dynamic_graph.edgeIndex[i][1]

            point_in_solution[index1] = 1
            point_in_solution[index2] = 1
            edge_count += 1

    node_count = sum(point_in_solution)

    # get result
    resulting_text = ''

    resulting_text += str(node_count) + '\n'

    for i in range(len(dynamic_graph.points)):

        if point_in_solution[i] == 1:
            resulting_text += str(dynamic_graph.points[i][0]) + ' ' \
                              + str(dynamic_graph.points[i][1]) + '\n'

    resulting_text += str(edge_count) + '\n'

    for i in range(len(dynamic_graph.edgeIndex)):
        if dynamic_graph_thresh[i] >= dynamic_threshold:
            edge = dynamic_graph.edgeIndex[i]
            resulting_text += str(edge[0]) + ' ' + str(edge[1]) + str(dynamic_graph_thresh[i]) + '\n'


f = open(output_filename, "w")
f.write(resulting_text)
f.close()

print("complete")
