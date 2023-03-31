import sys
from . import graph
import math

class DynamicTreeNode:

    def __init__(self, point, r: float, c: float):

        self.point = point
        self.reward = r
        self.cost = c
        self.initial_cost = c
        self.visitOnce = False
        self.isOldNode = False
        self.score = r  # temporary score
        self.total_cost = c  # temporary cost
        self.edges = list()

        self.index = -1

        self.drop_threshold = math.inf
        self.drop_index = None

    def add_edge(self, edge):
        self.edges.append(edge)

    # get node on the other end of the edge
    def get_other_node(self, edge):
        if self == edge.one: return edge.other
        return edge.one

    def get_unvisited_neigbor_count(self):
        count = 0
        for e in self.edges:
            if not self.get_other_node(e).visitOnce:
                count += 1
        return count

    def get_edge(self, otherNode):
        for e in self.edges:
            if self.get_other_node(e) == otherNode:
                return e

        print('Edge not found, error')
        return None

    def set_score(self):

        for e in self.edges:

            temp_node = self.get_other_node(e)
            if temp_node.visitOnce:
                # for dynamic tree
                e.set_score(temp_node, temp_node.score)
                e.set_cost(temp_node, temp_node.total_cost)
                self.score += temp_node.score
                self.total_cost += temp_node.total_cost


class DynamicTreeEdge:

    def __init__(self, one: DynamicTreeNode, other: DynamicTreeNode):
        self.one = one
        self.other = other
        self.one_to_other_score = None
        self.one_to_other_cost = None
        self.other_to_one_score = None
        self.other_to_one_cost = None

    def set_score(self, first_node, score):

        if first_node == self.one:
            self.one_to_other_score = score
        else:
            self.other_to_one_score = score

    def set_cost(self, first_node, total_cost):
        if first_node == self.one:
            self.one_to_other_cost = total_cost
        else:
            self.other_to_one_cost = total_cost

    def get_other_node(self, node: DynamicTreeNode):
        if node == self.one:
            return self.other
        return self.one


class DynamicTree:

    def __init__(self, points, edgeIndex, reward_list: list, cost_list: list):

        self.nodes = list()
        self.edges = list()

        # index doesn't change
        for i in range(len(points)):
            newNode = DynamicTreeNode(points[i], reward_list[i], cost_list[i])
            newNode.index = len(self.nodes)

            self.nodes.append(newNode)

        for i in range(len(edgeIndex)):
            firstIndex = edgeIndex[i][0]
            secondIndex = edgeIndex[i][1]
            edge = DynamicTreeEdge(cost_list[i], self.nodes[firstIndex], self.nodes[secondIndex])
            self.nodes[firstIndex].add_edge(edge)
            self.nodes[secondIndex].add_edge(edge)
            self.edges.append(edge)

    def duplicate(self):

        newTree = DynamicTree([],[],[],[])

        for node in self.nodes:
            newNode = DynamicTreeNode(node.point, node.reward, node.cost)
            newTree.add_node(newNode)

        for edge in self.edges:
            newEdge = DynamicTreeEdge(newTree.nodes[edge.one.index], newTree.nodes[edge.other.index])
            newTree.add_edge(newEdge)

        return newTree


    def to_colored_graph(self, tree_list: list):

        tree = tree_list[0]
        color_list = [0] * len(tree.edges)
        index = len(tree_list)

        for i in range(len(tree.edges)):
            color = 1.0
            for j in range(len(tree_list)):
                if not tree_list[j].hasEdge(tree.edges[i]):
                    color = float(j) / len(tree_list)
                    break
            color_list[i] = color

        graph = tree.to_graph()
        return graph, color_list

    def to_graph(self) -> graph.Graph:

        tree_graph = graph.Graph([], [])

        for node in self.nodes:
            tree_graph.points.append(node.point)

        for edge in self.edges:
            node1 = edge.one
            node2 = edge.other

            index1 = self.nodes.index(node1)
            index2 = self.nodes.index(node2)

            tree_graph.edgeIndex.append([index1, index2])

        return tree_graph

    def get_leaves(self):
        leaves_list = list()
        for node in self.nodes:

            if node.visitOnce:
                continue

            if node.get_unvisited_neigbor_count() < 2:
                leaves_list.append(node)

        return leaves_list

    def add_node(self, node: DynamicTreeNode):
        node.index = len(self.nodes)
        self.nodes.append(node)

    def add_edge(self, edge: DynamicTreeEdge):
        edge.one.add_edge(edge)
        edge.other.add_edge(edge)
        self.edges.append(edge)

    def describe(self):
        print("There are " + str(len(self.nodes)) + " nodes")
        print("There are " + str(len(self.edges)) + " edges")

    def hasEdge(self, edge: DynamicTreeEdge):
        point1 = edge.one.point
        point2 = edge.other.point
        for e in self.edges:
            if (e.one.point[0] == point1[0] and e.other.point[0] == point2[0]
                and e.one.point[1] == point1[1] and e.other.point[1] == point2[1])\
                    or (e.one.point[0] == point2[0] and e.other.point[0] == point2[0]
                        and e.one.point[1] == point2[1] and e.other.point[1] == point2[1]):
                return True
        return False

    def __str__(self):

        string = ""

        for node in self.nodes:
            string += ("Node " + str(node.point) + "\'s reward is " + str(node.reward) + "\n")
            string += ("Node " + str(node.point) + "\'s cost is " + str(node.cost) + "\n")

        return string
