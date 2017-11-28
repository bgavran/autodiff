def reverse_topo_sort(top_node):
    """
    Returns a list of nodes in the top_node subtree, sorted in the reverse topological order.

    :param top_node:
    :return: nodes iterator
    """

    def topo_sort_dfs(node, visited, topo_sort):
        if node in visited:
            return topo_sort
        visited.add(node)
        for n in node.children:
            topo_sort = topo_sort_dfs(n, visited, topo_sort)
        return topo_sort + [node]

    return reversed(topo_sort_dfs(top_node, set(), []))
