# Digital Image Processing
# Assignment-I (Question-1)
# Group-10

import numpy as np
# import cv2


# Node format to store pixel coordinate along with intensity in a graph i.e. (x, y):I
class PixelNode(object):
    def __init__(self, x, y, intensity):
        self.intensity = intensity
        self.x = x
        self.y = y

    def getx(self):
        return self.x

    def gety(self):
        return self.y

    def getIntensity(self):
        return self.intensity

    def __str__(self):
        return '(' + str(self.x) + ',' + str(self.y) + '):' + str(self.intensity)


# Storing source and destination for the edge
class Edge(object):
    def __init__(self, src, dest):
        self.src = src
        self.dest = dest

    def getSource(self):
        return self.src

    def getDestination(self):
        return self.dest


# Creating nodes and edges
class Digraph(object):
    def __init__(self):
        self.nodes = []
        self.edges = {}

    def addNode(self, node):
        if node in self.nodes:
            raise ValueError('Duplicate Node')
        else:
            self.nodes.append(node)
            self.edges[node] = []

    def addEdge(self, edge):
        src = edge.getSource()
        dest = edge.getDestination()
        if not (src in self.nodes and dest in self.nodes):
            raise ValueError('Node not in graph')
        self.edges[src].append(dest)

    def childrenOf(self, node):
        return self.edges[node]

    def __str__(self):
        result = ''
        for src in self.nodes:
            for dest in self.edges[src]:
                result = result + '(' + str(src.getx()) + ',' + str(src.gety()) + ') ->' + \
                         '(' + str(dest.getx()) + ',' + str(dest.gety()) + ')\n'
        return result[:-1]


# Graph
class Graph(Digraph):
    def addEdge(self, edge):
        Digraph.addEdge(self, edge)
        rev = Edge(edge.getDestination(), edge.getSource())
        Digraph.addEdge(self, rev)


# Printing the path
def printPath(path):
    result = ''
    for i in range(len(path)):
        result = result + str(path[i])
        if i != len(path) - 1:
            result = result + ' -> '
    return result


# BFS algorithm to find shortest path
def BFS(graph, start, end):
    counter = 0
    short_path = []
    xyz = []
    initPath = [start]
    pathQueue = [initPath]
    while len(pathQueue) != 0:
        tmpPath = pathQueue.pop(0)
        lastNode = tmpPath[-1]
        if lastNode == end:
            xyz.append(printPath(tmpPath))
            return tmpPath, len(xyz[0]), short_path, counter
        neighbors = graph.childrenOf(lastNode)
        for nextNode in neighbors:
            if nextNode not in tmpPath:
                newPath = tmpPath + [nextNode]
                pathQueue.append(newPath)
            if nextNode == end:
                counter += 1
                print(f"\nLength of Path #{counter}: {len(newPath) - 1}")
                print(f"Path #{counter}: {printPath(newPath)}")
                short_path.append(printPath(newPath))
    return None, None, None, None


# Function to identify 4-adjacency of a pixel
def sp_4adj(g, nodes, V):
    for i in range(len(nodes)):
        j = i + 1
        count = 0
        src_row = nodes[i].getx()
        src_col = nodes[i].gety()
        src_intensity = nodes[i].getIntensity()
        N4 = [[src_row, src_col + 1],
              [src_row, src_col - 1],
              [src_row - 1, src_col],
              [src_row + 1, src_col]]
        while j < len(nodes) or count == 4:
            dest_row = nodes[j].getx()
            dest_col = nodes[j].gety()
            dest_intensity = nodes[j].getIntensity()
            # print(f" for i = {i} and j = {j}\n x1={src_row},y1={src_col},x2={dest_row},y2={dest_col}")
            if src_intensity in V and dest_intensity in V:
                for point in N4:
                    if point[0] == dest_row and point[1] == dest_col:
                        g.addEdge(Edge(nodes[i], nodes[j]))
                        count += 1
            j += 1
    return g


# Function to identify 8-adjacency of a pixel
def sp_8adj(g, nodes, V):
    for i in range(len(nodes)):
        j = i + 1
        count = 0
        src_row = nodes[i].getx()
        src_col = nodes[i].gety()
        src_intensity = nodes[i].getIntensity()
        N8 = [[src_row, src_col + 1],
              [src_row, src_col - 1],
              [src_row - 1, src_col],
              [src_row + 1, src_col],
              [src_row + 1, src_col + 1],
              [src_row - 1, src_col + 1],
              [src_row + 1, src_col - 1],
              [src_row - 1, src_col - 1]]

        while j < len(nodes) or count == 8:
            dest_row = nodes[j].getx()
            dest_col = nodes[j].gety()
            dest_intensity = nodes[j].getIntensity()
            if src_intensity in V and dest_intensity in V:
                for point in N8:
                    if point[0] == dest_row and point[1] == dest_col:
                        g.addEdge(Edge(nodes[i], nodes[j]))
                        count += 1
            j += 1
    return g


# Function to identify m-adjacency of a pixel
def sp_madj(g, nodes, V):
    for i in range(len(nodes)):
        j = i + 1
        src_row = nodes[i].getx()
        src_col = nodes[i].gety()
        src_intensity = nodes[i].getIntensity()
        src_N4 = [[src_row, src_col + 1],
                  [src_row, src_col - 1],
                  [src_row - 1, src_col],
                  [src_row + 1, src_col]]
        while j < len(nodes):
            dest_row = nodes[j].getx()
            dest_col = nodes[j].gety()
            dest_intensity = nodes[j].getIntensity()
            a = 0
            b = 0
            if src_intensity in V and dest_intensity in V:
                for point in src_N4:
                    if point[0] == dest_row and point[1] == dest_col:
                        g.addEdge(Edge(nodes[i], nodes[j]))
                        a = 1
                if a != 1:
                    src_ND = [[src_row + 1, src_col + 1],
                              [src_row - 1, src_col + 1],
                              [src_row + 1, src_col - 1],
                              [src_row - 1, src_col - 1]]
                    q_N4 = [[dest_row, dest_col + 1],
                            [dest_row, dest_col - 1],
                            [dest_row - 1, dest_col],
                            [dest_row + 1, dest_col]]
                    for p in src_ND:
                        if p[0] == dest_row and p[1] == dest_col:
                            b = 1
                    if b == 1:
                        intersection = []
                        for src in src_N4:
                            if src in q_N4:
                                intersection.append(src)
                        l = len(intersection)
                        count = 0
                        while len(intersection) != 0:
                            tmp_node = intersection.pop(0)
                            for node in nodes:
                                x = node.getx()
                                y = node.gety()
                                if tmp_node[0] == x and tmp_node[1] == y:
                                    intensity = node.getIntensity()
                                    if intensity not in V:
                                        count += 1
                        if count == l:
                            g.addEdge(Edge(nodes[i], nodes[j]))
            j += 1
    return g


# Creating start and end node from the pixel coordinates
def create_pixel_map(I, x1, y1, x2, y2, path_type, V):
    size = I.shape
    nodes = []
    start = None
    end = None

    # Create Nodes out of the coordinates
    for row in range(size[0]):
        for col in range(size[1]):
            nodes.append(PixelNode(row, col, I[row, col]))
            if row == x1 and col == y1:
                start = nodes[-1]
            if row == x2 and col == y2:
                end = nodes[-1]
            # print(start, end)    

    # Create Undirected Edges for different path types    
    g = Graph()
    for n in nodes:
        g.addNode(n)

    if path_type == '4':
        return (sp_4adj(g, nodes, V), start, end)
    elif path_type == '8':
        return (sp_8adj(g, nodes, V), start, end)
    elif path_type == '10':
        return (sp_madj(g, nodes, V), start, end)
    else:
        return (None, None, None)


# Function to find required paths
def find_paths(I, x1, y1, x2, y2, V, path_type):
    (g, start, end) = create_pixel_map(I, x1, y1, x2, y2, path_type, V)
    if g == None:
        print('Incorrect Path Type entered.')
        return None
    if start == None or end == None:
        print('Incorrect Start or End coordinates entered.')
        return None
    print(f"Path Type: {path_type}")
    # print("Graph: \n",g)
    sp, short_len, short_path, counter = BFS(g, start, end)
    if sp == None:
        print('\nNo Path Found.')
    else:
        print(f"\nTotal {counter} {path_type}-path exists.")
        print("\nLength of Shortest Path:", len(sp) - 1)
        for i in range(len(short_path)):
            if len(short_path[i]) == short_len:
                print(f"Shortest Path #{i + 1}: {short_path[i]}")

# Main function                         # parameter description
def main():
    # I = cv2.imread('image.jpg', 0)    # 0 for grayscale image
    # I = cv2.resize(I,(50,50))
    # cv2.imshow("Image", I)
    # cv2.waitKey(0)

    I = np.array([[1, 0, 3, 2, 4],      # An image as 2D matrix
                  [4, 3, 4, 0, 2],
                  [2, 2, 1, 3, 0],
                  [2, 4, 0, 2, 3],
                  [3, 2, 4, 1, 0]])
    x1 = 3                              # (x1, y1) and (x2, y2) coordinates of 2 points
    y1 = 0
    x2 = 1
    y2 = 4
    V = [4, 2]                          # set V as an array
    path_type = '10'                    # type of path ( = 4, 8, 10 (10 for m-path))
    
    # Function call
    find_paths(I, x1, y1, x2, y2, V, path_type)


if __name__ == '__main__':
    main()