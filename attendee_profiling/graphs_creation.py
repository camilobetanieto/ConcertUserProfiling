import os
import pandas as pd
import networkx as nx
import sys

def read_file(file_path):
    try:
        # Check if directory exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Directory {file_path} does not exist!")

        # Read CSV of user scores and display info
        print(f"\nReading {file_path}:")
        print("-" * 50)            
            
        df = pd.read_csv(file_path)
        df.info()
        print("-" * 50)

        return df
    
    except Exception as e:
        print(e)



def build_event_attendance_network_durations(df):
    """
    Builds a bipartite network where nodes are attendees (uid) and events (event_title).
    Attendee nodes will have an empty label, while event nodes will be labeled with their event name.
    The edge weight is the sum of duration_stop_mins spent by an attendee at that event.
    """
    G = nx.Graph()

    # Add nodes for attendees with empty label and events with their event name as label.
    attendees = df['uid'].unique()
    events = df['event_title'].unique()
    for uid in attendees:
        G.add_node(uid, bipartite=0, label="", type='attendee')  # No label for attendee
    for event in events:
        G.add_node(event, bipartite=1, label=event, type='event')  # Label equals event name

    # For each row, add or update the edge between the attendee and the event.
    for _, row in df.iterrows():
        uid = row['uid']
        event = row['event_title']
        duration = row['time_at_event_mins']
        if G.has_edge(uid, event):
            G[uid][event]['weight'] += duration
        else:
            G.add_edge(uid, event, weight=duration)

    return G


def build_event_attendance_network_scores(df):
    """
    Builds a bipartite network where nodes are attendees (uid) and events (event_title).
    Attendee nodes will have an empty label, while event nodes will be labeled with their event name.
    The edge weight is the sum of adj_score spent by an attendee at that event.
    """
    G = nx.Graph()

    # Add nodes for attendees with empty label and events with their event name as label.
    attendees = df['uid'].unique()
    events = df['event_title'].unique()
    for uid in attendees:
        G.add_node(uid, bipartite=0, label="", type='attendee')  # No label for attendee
    for event in events:
        G.add_node(event, bipartite=1, label=event, type='event')  # Label equals event name

    # For each row, add or update the edge between the attendee and the event.
    for _, row in df.iterrows():
        uid = row['uid']
        event = row['event_title']
        duration = row['adj_score']
        if G.has_edge(uid, event):
            G[uid][event]['weight'] += duration
        else:
            G.add_edge(uid, event, weight=duration)
    return G

def display_network_info(G, title="Network"):
    """
    Displays basic network info.
    """
    print(f"\n{title} Info:")
    print("-" * 50)
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print("-" * 50)

def save_network_gexf(G, output_path):
    """
    Saves the network G in GEXF format.
    """
    try:
        nx.write_gexf(G, output_path)
        print(f"Network saved to {output_path}")
    except Exception as e:
        print(f"Error saving network: {e}")


if __name__ == "__main__":

    # Read the script parameters
    print("Script name:", sys.argv[0])

    # Paths of the directories to read and write are parameters 1 and 2
    if len(sys.argv) > 1:
        path_tables_for_description = sys.argv[1]
        path_graph_files = sys.argv[2]

        print(f'Reading data from {path_tables_for_description}')
        print(f'Path to save the networks: {path_graph_files}')
    else:
        raise Exception("No parameters provided. The paths should be passed as follows:\n"
        "Usage: python script_name.py <path_to_tables_for_description> <path_to_gephi_networks>\n"
        )

   
    # Build the event attendance network based on one of your CSV files.
    csv_file = os.path.join(path_tables_for_description, "user_event_scores_durations_night.csv")
    if os.path.exists(csv_file):

        # Read the file
        user_event_scores = read_file(csv_file)

        # Segment the dataset for each of the nights
        user_event_scores_list = [user_event_scores.loc[user_event_scores['tid']==night] for night in (1,2)]

        for i, user_event_scores in enumerate(user_event_scores_list):
            # Build the network based on the total time spent at the events (for each night)
            event_net_durations = build_event_attendance_network_durations(user_event_scores)
            display_network_info(event_net_durations, f"Event Attendance Network - Durations (night {i+1})")

            # Build the network based on the score assigned to the events
            event_net_scores = build_event_attendance_network_scores(user_event_scores)
            display_network_info(event_net_scores, f"Event Attendance Network - Scores (night {i+1})")

            # Create the general file names (without the extension)
            output_file_durations = os.path.join(path_graph_files, f"event_attendance_network_durations_n{i+1}.gexf")
            output_file_scores = os.path.join(path_graph_files, f"event_attendance_network_scores_n{i+1}.gexf")
        
            # Save the durations network (GEXF format)
            save_network_gexf(event_net_durations, output_file_durations)

            # Save the scores network (GEXF format)
            save_network_gexf(event_net_scores, output_file_scores)
    else:
        print(f"{csv_file} does not exist.")