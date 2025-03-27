import pandas as pd
import numpy as np
import networkx as nx

class MBTA():
    def __init__(self, path: str = 'data\\MBTA_Rapid_Transit_Stop_Distances.csv'):
        df = pd.read_csv(path)
        self._create_graph(df)

    def fastest_route(self, station1: str, station2: str):
        """Calculate fastest route between two stations"""
        return nx.shortest_path(self.gr, station1, station2)

    def _create_graph(self, df: pd.DataFrame):
        gr = nx.Graph()
        nodes = set()  # We use sets because there are no duplicates
        edges = [] # not a set bc of floats -- floats are dangerous

        existing = set()

        for idx, row in df.iterrows():
            nodes.add(row['from_stop_name'])
            nodes.add(row['to_stop_name'])

            #edge = (row['from_stop_name'], row['to_stop_name'], row['from_to_miles'])

            # this will be weighted by distance 

            if ((row['from_stop_name'], row['to_stop_name']) not in existing and (row['from_stop_name'],row['to_stop_name']) not in existing):
                existing.add((row['from_stop_name'], row['to_stop_name']))
                edges.append((row['from_stop_name'], row['to_stop_name'], row['from_to_miles']))

        gr.add_nodes_from(nodes)
        gr.add_weighted_edges_from(edges) # defaults from_to_miles as the weight

        self.gr = gr

if __name__ == '__main__':
    mbta = MBTA()
    print(mbta.fastest_route('Copley','Airport'))


