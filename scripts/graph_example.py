import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

# Make a list of nodes and edges

nodes = list(range(9)) # makes 0-8; node = numbers

edges = [(1, 0), (2, 1), (3, 2), (4, 1), (5, 0),
         (0, 5), (6, 3), (7, 3), (8, 0)] # represents connections

# Directed graph; if you ever want undirected do nx.Graph()

gr = nx.DiGraph()

gr.add_nodes_from(nodes)
gr.add_edges_from(edges)

# pagerank

pr = nx.pagerank(gr,max_iter=1000)

pos = nx.spring_layout(gr)



# Pretend each connection is a spring

nx.draw_networkx_nodes(gr, pos=pos,node_size=[pr[node]*300 for node in nodes], node_color='green')
nx.draw_networkx_edges(gr, pos=pos)
plt.show()