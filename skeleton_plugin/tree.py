import sys
from . import graph

class TreeNode:

    def __init__(self, point, r: float):

        self.point = point
        self.reward = r
        self.visitOnce = False
        self.visitTwice = False
        self.solutionChecked = False
        self.score = r #temporary score
        self.edges = list()

    def add_edge(self, edge):
        self.edges.append(edge)

    #get node on the other end of the edge
    def get_other_node(self,edge):
        if self == edge.one: return edge.other
        return edge.one

    def get_unvisited_neigbor_count(self):
        count = 0
        for e in self.edges:
            if not self.get_other_node(e).visitOnce:
                count += 1
        return count

    def get_edge_cost(self,otherNode):
        for e in self.edges:
            if self.get_other_node(e) == otherNode:
                return e.cost

        print('Edge not fount, error')

    def set_score(self):

        for e in self.edges:

            temp_node = self.get_other_node(e)
            if temp_node.visitOnce:
                self.score += max(0, temp_node.score+e.cost)

    def update_score(self):

        for e in self.edges:

            temp_node = self.get_other_node(e)

            if temp_node.visitTwice:
                temp_score = self.score - max(0, self.score + e.cost) + max(0, temp_node.score + e.cost)
                self.score = max(temp_score, self.score)


class TreeEdge:

    def __init__(self,c: float, one: TreeNode, other: TreeNode):
        self.one = one
        self.other = other
        self.cost = c

    '''
    def get_other_node(self, node: TreeNode):
        if node == self.one:
            return self.other
        return self.one
    '''

class Tree:

    def __init__(self, points, edgeIndex, reward_list:list, cost_list:list):

        self.nodes = list()
        self.edges = list()

        for i in range(len(points)):
            self.nodes.append(TreeNode(points[i],reward_list[i]))

        for i in range(len(edgeIndex)):
            firstIndex = edgeIndex[i][0]
            secondIndex = edgeIndex[i][1]
            edge = TreeEdge(cost_list[i],self.nodes[firstIndex], self.nodes[secondIndex])
            self.nodes[firstIndex].add_edge(edge)
            self.nodes[secondIndex].add_edge(edge)
            self.edges.append(edge)

    def get_leaves(self):
        leaves_list = list()
        for node in self.nodes:

            if node.visitOnce:
                continue

            if node.get_unvisited_neigbor_count() < 2:
                leaves_list.append(node)

        return leaves_list

    def add_node(self, node: TreeNode):
        self.nodes.append(node)

    def add_edge(self, edge: TreeEdge):
        edge.one.add_edge(edge)
        edge.other.add_edge(edge)
        self.edges.append(edge)

    def to_graph(self):
        points = list()
        temp_nodes = list()
        edgeIndex = list()

        for node in self.nodes:
            temp_nodes.append(node)
            points.append(node.point)

        for edge in self.edges:
            curr = []
            curr.append(temp_nodes.index(edge.one))
            curr.append(temp_nodes.index(edge.other))
            edgeIndex.append(curr)

        return graph.Graph(points, edgeIndex)
