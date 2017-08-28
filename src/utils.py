from graphviz import Digraph
from core.ops import *


class MyDigraph(Digraph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_of_added_nodes = set()
        self.sg = None
        self.parent = None

    def node(self, name, label=None, _attributes=None, **attrs):
        super().node(name, label, _attributes, **attrs)
        self.set_of_added_nodes.add(name)

    def subgraph(self, graph=None, name=None, comment=None,
                 graph_attr=None, node_attr=None, edge_attr=None, body=None):

        if graph is None:
            kwargs = {'name': name, 'comment': comment,
                      'graph_attr': graph_attr, 'node_attr': node_attr,
                      'edge_attr': edge_attr, 'body': body}
            return MySubgraphContext(self, kwargs)
        else:
            super().subgraph(graph, name, comment, graph_attr, node_attr, edge_attr, body)

    def add_node(self, node):
        attributes = {"label": node.name,
                      "color": MyDigraph.get_color(node),
                      "shape": MyDigraph.get_shape(node)}

        self.node(str(node.id), **attributes)

    def add_edge(self, child, parent):
        child = child.get_node_for_graph()
        parent = parent.get_node_for_graph()
        self.edge(str(child.id),
                  str(parent.id),
                  **{"style": MyDigraph.get_edge_style(parent)})

    @staticmethod
    def get_color(node):
        if isinstance(node, CompositeOperation):
            return "aquamarine3"
        if isinstance(node, Variable):
            return "indianred1"
        elif isinstance(node, Constant):
            return "gray"
        else:
            return "lightblue"

    @staticmethod
    def get_shape(node):
        if isinstance(node, CompositeOperation):
            return "doubleoctagon"
        if isinstance(node, Variable) or isinstance(node, Constant):
            return "box"
        else:
            return "oval"

    @staticmethod
    def get_edge_style(node):
        from core.ops import Grad
        if isinstance(node, Grad):
            return "dashed"
        else:
            return "filled"

    @staticmethod
    def get_edge_arrow(node):
        return "normal"


class MySubgraphContext:
    """Return a blank instance of the parent and add as subgraph on exit."""

    def __init__(self, parent, kwargs):
        self.parent = parent
        self.graph = parent.__class__(**kwargs)

    def __enter__(self):
        return self.graph

    def __exit__(self, type_, value, traceback):
        if type_ is None:
            self.parent.subgraph(self.graph)
            self.parent.set_of_added_nodes.update(self.graph.set_of_added_nodes)

            self.parent.sg = self.graph
            self.graph.parent = self.parent


def plot_comp_graph(root_node, view=False):
    graph = MyDigraph("Computational graph", filename="comp_graph", engine="dot")
    graph.attr(size="6,6")
    graph.node_attr.update(color='lightblue2', style="filled")
    graph.graph_attr.update(rankdir="BT")

    root_node.add_node_subgraph_to_plot_graph(graph)

    graph.render(view=view)
