from graphviz import Digraph

import core


def plot_comp_graph(node, show_all_ops=True):
    graph = Digraph("Computational graph", filename="comp_graph", engine="dot")
    graph.attr(size="6,6")
    graph.node_attr.update(color='lightblue2', style="filled")
    graph.graph_attr.update(rankdir="BT")

    nodes = [(str(i),
              {"label": node.find_node_by_id(i).name,
               "color": "indianred1" if isinstance(node.find_node_by_id(i), core.ops.Variable) else "lightblue",
               "shape": "box" if isinstance(node.find_node_by_id(i), core.ops.Variable) else "oval"})
             for i in node.all_ids]
    edges = edges_from_tree(node, [])

    graph = add_nodes(graph, nodes)
    graph = add_edges(graph, edges)

    graph.view()


def edges_from_tree(graph, curr_edge_list):
    for child in graph.children:
        style = "filled"
        if isinstance(graph, core.ops.Grad):
            style = "dashed"

        curr_edge_list.append(((str(child.id), str(graph.id)), {"style": style}))
        curr_edge_list = edges_from_tree(child, curr_edge_list)
    return curr_edge_list


def add_nodes(graph, nodes):
    for n in nodes:
        if isinstance(n, tuple):
            graph.node(n[0], **n[1])
        else:
            graph.node(n)
    return graph


def add_edges(graph, edges):
    for e in edges:
        if isinstance(e[0], tuple):
            graph.edge(*e[0], **e[1])
        else:
            graph.edge(*e)
    return graph
