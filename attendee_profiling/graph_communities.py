import networkx as nx
from networkx.algorithms import bipartite
from sknetwork.clustering import Louvain, get_modularity
from scipy import sparse
import numpy as np
import pandas as pd

from attendee_profiling import constants


def bipartite_louvain_multiresolutions(biadjacency_matrix, resolutions):
    results_list = []
    for resolution in resolutions:
        # Create the louvain object and execute the algorithm
        louvain = Louvain(resolution=resolution,
                          random_state=constants.RANDOM_SEED)
        louvain.fit(biadjacency_matrix)

        # Get the community labels
        labels_row = louvain.labels_row_
        labels_col = louvain.labels_col_

        # Get the modularity
        modularity = get_modularity(biadjacency_matrix,
                                    labels_row, labels_col)

        # Count communities:
        num_comms_row = len(np.unique(labels_row))
        num_comms_col = len(np.unique(labels_col))

        # Append the results
        results_list.append({
            'resolution': resolution,
            'labels_attendees': labels_row,
            'labels_events': labels_col,
            'num_attendee_comms': num_comms_row,
            'modularity': modularity
            })
        
        results_df = pd.DataFrame(results_list).sort_values(by='modularity', ascending=False)
        
    return results_df


def add_labels_to_graph(graph, attendees_nodes, event_nodes, labels_attendees, labels_events):
    for i, node in enumerate(attendees_nodes):
        graph.nodes[node]['community'] = int(labels_attendees[i])

    for i, node in enumerate(event_nodes):
        graph.nodes[node]['community'] = int(labels_events[i])

    return graph