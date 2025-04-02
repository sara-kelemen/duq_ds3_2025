import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def read_data_fb(path: str):
    """Read in the data"""
    data = pd.read_csv(path, sep=' ', header = None, names=['node1','node2'])

    node1 = data['node1'].tolist()
    node2 = data['node2'].tolist()

    out = [] #init empty

    # assert means make sure this is true!
    assert len(node1) == len(node2) # throws an error if it is not true

    for i in range(len(node1)):
        out.append((node1[i],node2[i]))

    return out

def read_data_fb_v2(path: str):
    """Read in the data using zip"""
    data = pd.read_csv(path, sep=' ', header = None, names=['node1','node2'])
    return zip(data['node1'],data['node2'])

# Unique groups, plot, and make them different colors

def graph_by_importance(edges):
    gr= nx.DiGraph()
    gr.add_edges_from(edges)

    pr = nx.pagerank(gr,max_iter=1000)  # importance
    pos = nx.spring_layout(gr)

    nx.draw_networkx(
        gr,
        pos=pos
    )

    plt.show()

def graph_unique(edges):
    gr = nx.Graph()
    gr.add_edges_from(edges)

    connected_components = list(nx.connected_components)
    pos = nx.spring_layout(gr)
    nx.draw_networkx_edges()

if __name__ == '__main___':
    path = 'data\\facebook_combined.txt'
    edges = read_data_fb(path)   # or read_data_fb_v2(path)
    graph_unique(edges)


"""This code is largely unfinished"""