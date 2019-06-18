# last node id
nodeId = 0


class Node:
    """
    Graph node.
    """

    def __init__(self):
        global nodeId
        self.id = nodeId
        nodeId += 1
        self.outputLinks = []
        self.groupId = 0
        self.inputCount = 0

    def addOutputs(self, nodeList):
        """
        Adds list of output nodes.
        """
        for node in nodeList:
            node.inputCount += 1
            self.outputLinks.append(node)

    def __repr__(self):
        return f"{self.id}"

    def __str__(self):
        return f"{self.id}"


def groupNodes(nodeList):
    """
    Topological sort with grouping. Returns list of Node groups.
    """

    # finding root nodes
    BFSList = [node for node in nodeList if node.inputCount == 0]

    # counting groups
    maxGroupId = 0

    # traversing nodes to assign group ids to them
    while(len(BFSList) > 0):
        # next step of BFS algorithm
        nextList = []
        for node in BFSList:
            # nodes which are further from root nodes will get higher group ids
            nextGroupId = node.groupId + 1
            for outputNode in node.outputLinks:
                if outputNode.groupId < nextGroupId:
                    # if group id of next node is lower than it should be, it is updated
                    outputNode.groupId = nextGroupId
                    maxGroupId = nextGroupId if nextGroupId > maxGroupId else maxGroupId
                if outputNode not in nextList:
                    nextList.append(outputNode)
        # going to the next step
        BFSList = nextList

    # after group ids are assigned, returning groups of nodes
    groups = []
    for i in range(maxGroupId + 1):
        group = [node for node in nodeList if node.groupId == i]
        groups.append(group)
    return groups


# module test
if __name__ == "__main__":
    node0 = Node()
    node1 = Node()
    node2 = Node()
    node3 = Node()
    node4 = Node()
    node5 = Node()
    node6 = Node()
    node0.addOutputs([node1])
    node1.addOutputs([node2])
    node2.addOutputs([node5])
    node3.addOutputs([node4])
    node4.addOutputs([node5])
    node5.addOutputs([node6])
    nodeList = [node0, node1, node2, node3, node4, node5, node6]
    for group in groupNodes(nodeList):
        print(group)
