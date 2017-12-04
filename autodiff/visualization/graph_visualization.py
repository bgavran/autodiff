import numbers
from graphviz import Digraph
from ..core.node import Variable


def plot_comp_graph(top_node, view=False, name="comp_graph"):
    print("\nPlotting...")
    graph = MyDigraph("Computational graph", filename=name, engine="dot")
    graph.attr(size="6,6")
    graph.node_attr.update(color='lightblue2', style="filled")
    graph.graph_attr.update(rankdir="BT")

    graph.add_node_subgraph_to_plot_graph(top_node)

    graph.render(view=view, cleanup=True)


class MyDigraph(Digraph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.added_nodes = set()

    @staticmethod
    def id_str(node):
        return str(node.id)

    def add_node(self, node, root_graph=None):
        if root_graph is None:
            root_graph = self
        super().node(MyDigraph.id_str(node),
                     label=str(node),
                     color=MyDigraph.get_color(node),
                     shape=MyDigraph.get_shape(node))
        root_graph.added_nodes.add(MyDigraph.id_str(node))

    def add_edge(self, child, parent):
        self.edge(MyDigraph.id_str(child),
                  MyDigraph.id_str(parent),
                  **{"style": "filled"})

    @staticmethod
    def get_color(node):
        if isinstance(node, Variable):
            # better way to figure out the coloring?
            if isinstance(node.value, numbers.Number) and node.value == 1 and node.name[-4:] == "grad":
                return "gray"
            return "indianred1"
        else:
            return "lightblue"

    @staticmethod
    def get_shape(node):
        if isinstance(node, Variable):
            return "box"
        else:
            return "oval"

    def add_node_with_context(self, node, ctx, root_graph=None):
        """
        Add just the node (not the connections, not the children) to the respective subgraph
        """
        if root_graph is None:
            root_graph = self
        if len(ctx):
            with self.subgraph(name="cluster" + ctx[0]) as subgraph:
                subgraph.attr(color="blue")
                subgraph.attr(label=ctx[0].split("_")[0])

                subgraph.add_node_with_context(node, ctx[1:], root_graph=self)
        else:
            self.add_node(node, root_graph)

    def add_node_subgraph_to_plot_graph(self, top_node):
        if MyDigraph.id_str(top_node) not in self.added_nodes:
            self.add_node_with_context(top_node, top_node.context_list)

            # Add connections to children
            for child in top_node.children:
                self.add_edge(child, top_node)

            # Make each of the children do the same, but skip duplicates
            for child in set(top_node.children):
                self.add_node_subgraph_to_plot_graph(child)
