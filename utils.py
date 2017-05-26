from graphviz import Digraph
from computational_graph import CompositeOperation


def plot_comp_graph(node):
    graph = Digraph("Computational graph", filename="comp_graph", engine="dot")
    graph.attr(size="6,6")
    graph.node_attr.update(color='lightblue2', style="filled")
    graph.graph_attr.update(rankdir="BT")

    def exists_in_graph(node_id, graph):
        nodes = [elem.split("->") for elem in graph.body]
        list_of_elems = []
        for elems in nodes:
            if len(elems) > 1:
                list_of_elems.append(int(elems[1]))

        return node_id in list_of_elems

    def add_recursively(node):
        if isinstance(node, CompositeOperation):
            node = node.out
        color = "indianred1"
        if hasattr(node, "children"):
            color = "lightblue"
            for child in node.children:
                graph.edge(str(child.id), str(node.id))
                if not exists_in_graph(child.id, graph):
                    add_recursively(child)
        graph.node(str(node.id), label=node.name, color=color)

    add_recursively(node)

    graph.view()
