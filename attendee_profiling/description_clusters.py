import seaborn as sns
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import numpy as np
import skmob
import os
from IPython.display import display

from attendee_profiling import config, utils

# Assign the path to read the table related to the event rankings.
TABLES_DESCRIPTION_PATH = config.TABLES_DESCRIPTION_PATH

# Read the sequence of stops table
user_event_scores_durations_night = pd.read_csv(os.path.join(TABLES_DESCRIPTION_PATH,'user_event_scores_durations_night.csv'))


# General settings for plotting
sns.set_theme('paper', style="dark")
matplotlib_palette = 'tab10'
plotly_palette = px.colors.qualitative.D3


#----------------------------------------------------------------------------------------


def plot_arrival_departure(df, clusters_col, palette=matplotlib_palette):
    """
    Plots density distributions of arrival_festival_mins and departure_festival_mins 
    in separate subplots, differentiated by cluster labels, with vertical titles on the right.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    clusters_col (str): The column name corresponding to the cluster labels.
    """
    fig, axes = plt.subplots(2, 1, figsize=(11, 4.5), sharex=True)
    time_types = ['arrival_festival_mins', 'departure_festival_mins']
    titles = ['Arrival time', 'Departure time']

    for i, time_type in enumerate(time_types):
        ax = axes[i]
        
        # KDE Plot
        sns.kdeplot(data=df, x=time_type, hue=clusters_col, fill=True, alpha=0.5, linewidth=1.2, palette=palette, ax=ax)
        ax.set_ylabel("Density")
        
        # Remove upper and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Add vertical title on the right
        ax.text(1.0, 0.5, titles[i], transform=ax.transAxes, fontsize=12, 
                rotation=-90, ha='left', va='center')

    axes[-1].set_xlabel("Minutes from the festival start") 
    
    plt.tight_layout()
    plt.show()





def plot_violin_box_by_cluster(df, clusters_col, interest_col, x_label=None, y_label=None, palette=plotly_palette):
    """
    Plots a combined violin and box plot for a column of interest, differentiated by cluster labels.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    clusters_col (str): The column name corresponding to the cluster labels.
    interest_col (str): The column name corresponding to the column of interest to plot.
    """
    # Create a plotly figure with both violin and box plot
    fig = px.violin(df, x=clusters_col, y=interest_col, box=True, points="outliers", 
                    title=f'Violin and Box Plot of {interest_col} by {clusters_col}', 
                    color=clusters_col, 
                    color_discrete_sequence=palette, 
                    category_orders={clusters_col: sorted(df[clusters_col].unique())})

    if not x_label:
        x_label = clusters_col

    if not y_label:
        y_label = interest_col
    
    # Update layout for better clarity
    fig.update_layout(
        yaxis_title=y_label,
        xaxis_title=x_label,
        title=dict(font=dict(size=15)),
        xaxis=dict(title=dict(font=dict(size=14))),
        yaxis=dict(title=dict(font=dict(size=14))),
        showlegend=False
    )

    fig.show()





def plot_scatter_with_regression(df, x_col, y_col, clusters_col, x_label=None, y_label=None, palette=matplotlib_palette):
    """
    Plots a scatter plot with linear regression lines for each cluster.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    x_col (str): The column name for the x-axis.
    y_col (str): The column name for the y-axis.
    clusters_col (str): The column name for the cluster labels.
    palette (str): The color palette for the clusters (default is 'viridis').
    """
    # Create the scatter plot with linear regression lines for each cluster
    sns.lmplot(data=df, x=x_col, y=y_col, hue=clusters_col, aspect=1.8, height=4,
               palette=palette, scatter_kws={'alpha': 0.7}, line_kws={'linewidth': 2})

    
    
    # Set axis labels and title
    if not x_label:
        x_label = x_col.replace('_', ' ').capitalize()

    if not y_label:
        y_label = y_col.replace('_', ' ').capitalize()


    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f'{x_col.replace("_", " ").capitalize()} vs. {y_col.replace("_", " ").capitalize()}')
    
    # Show the plot
    plt.show()




def user_event_scores_with_clusters(trajectory_general_measures_df, clusters_col, sonar_type='night'):
    """
    Merges user event scores with cluster information.

    Parameters:
    ----------
    trajectory_general_measures_df : pandas.DataFrame
        A DataFrame containing general trajectory measures, including user and trajectory IDs (uid, tid) 
        and a clustering column.
    clusters_col : str
        The column name representing cluster labels in `trajectory_general_measures_df`.
    sonar_type : str, optional
        The sonar type determining which user event scores dataset to use. Default is 'night'.

    Returns:
    -------
    pandas.DataFrame
        A DataFrame containing user event scores merged with cluster labels.

    """
    if sonar_type == 'night':
        user_event_scores_durations_df = user_event_scores_durations_night.copy()

    user_event_scores_durations_df = pd.merge(user_event_scores_durations_df,
                                                trajectory_general_measures_df[['uid', 'tid',clusters_col]],
                                                on=['uid','tid'],
                                                how='left')
    return user_event_scores_durations_df





def plot_rankings_by_cluster(trajectory_general_measures_df, clusters_col, show_plot=True, path_to_save=None, sonar_type='night'):
    """
    Plots event rankings for each cluster using `utils.plot_rankings()`.

    Parameters:
    ----------
    trajectory_general_measures_df : pandas.DataFrame
        A DataFrame containing general trajectory measures, including user and trip IDs (uid, tid) 
        and a clustering column.
    clusters_col : str
        The column name representing cluster labels in `trajectory_general_measures_df`.
    show_plot : bool, optional
        Whether to display the plots. Default is True.
    path_to_save : str, optional
        Directory path where ranking plots will be saved as PDFs. If None, plots are not saved. Default is None.
    sonar_type : str, optional
        The sonar type determining which user event scores dataset to use. Default is 'night'.

    Returns:
    -------
    None
        Displays and/or saves ranking plots for each cluster.

    """
    user_event_scores_clusters_df = user_event_scores_with_clusters(trajectory_general_measures_df,
                                                                    clusters_col,
                                                                    sonar_type=sonar_type)
    user_event_scores_clusters_grouped = user_event_scores_clusters_df.groupby(by=clusters_col)

    for cluster_label, group_df in user_event_scores_clusters_grouped:
        event_final_scores_cluster = utils.compute_event_final_scores(user_event_scores_durations_df=group_df)

        if path_to_save:
            clusters_col_split =  clusters_col.split('_')[0:-1]
            cluster_method = '_'.join(clusters_col_split)
            filename_to_save = os.path.join(path_to_save,f'rankings_{cluster_method}_{int(cluster_label)}.pdf')
        else:
            filename_to_save = None

        utils.plot_rankings(event_final_scores_cluster, show_plot=show_plot, filename_to_save=filename_to_save)




def  plot_record_h3counts_by_cluster(trajectories_df, clusters_col, sonar_type='night', path_to_save=None):
    """
    Plots and saves maps of record counts per H3 cell for each cluster in a given trajectory dataset.

    The maps are saved as HTML files.

    Parameters:
    -----------
    trajectories_df : pandas.DataFrame
        A DataFrame containing trajectory data, including H3 cell information and cluster labels.
    
    clusters_col : str
        The name of the column in `trajectories_df` that contains cluster labels.
    
    sonar_type : str, optional (default='night')
        The type of sonar data to be used in plotting.
    
    path_to_save : str, required
        The directory path where the generated maps should be saved.
        If not provided, an exception is raised.

    Notes:
    ------
    - The function does not return the maps to avoid visualization issues in notebooks.
    - The saved filenames follow the format: 
      `cell_total_observations_{cluster_method}_{cluster_label}.html`

    """
    trajectories_grouped = trajectories_df.groupby(clusters_col)

    for cluster_label, group_df in trajectories_grouped:

        records_per_cell_cluster = utils.trajectories_to_h3_counts_gdf(trajectories_df=group_df,
                                                                       count_type='records',
                                                                       window_duration=None,
                                                                       sonar_type=sonar_type)
        
        # Extract the cluster method for the plot caption
        clusters_col_split =  clusters_col.split('_')[0:-1]
        cluster_method = '_'.join(clusters_col_split)

        # Convert to int for clarity when printing and saving (could be float)
        cluster_label = int(cluster_label)

        # Create the filename to save if needed
        if path_to_save:
            filename_to_save = os.path.join(path_to_save,f'cell_total_observations_{cluster_method}_{cluster_label}.html')
        else:
            raise Exception('`path_to_save` must be provided to save the plots')

        # Plot the counts. Maps are saved but not returned as there would be problems when visuaizing them within a notebook
        utils.plot_counts_per_cell(sonar_type=sonar_type,
                                   h3_cells_counts=records_per_cell_cluster,
                                   caption=f"Observation count per H3 Cell ({cluster_method}_{cluster_label})",
                                   filename_to_save=filename_to_save,
                                   return_map=False)





def plot_user_h3counts_timeline_by_cluster(trajectories_df, clusters_col, window_duration='5min', sonar_type='night', path_to_save=None):
    """
    Plots and saves timeline maps of unique user counts per H3 cell for each cluster in a trajectory dataset.

    The maps are saved as HTML files.

    Parameters:
    -----------
    trajectories_df : pandas.DataFrame
        A DataFrame containing trajectory data, including H3 cell information and cluster labels.

    clusters_col : str
        The name of the column in `trajectories_df` that contains cluster labels.

    window_duration : str, optional (default='5min')
        The time window duration for aggregating user counts (e.g., '5min', '10min').

    sonar_type : str, optional (default='night')
        The type of sonar data to be used in plotting.

    path_to_save : str, required
        The directory path where the generated maps should be saved.
        If not provided, an exception is raised.

    Notes:
    ------
    - The function does not return the maps to avoid visualization issues in notebooks.
    - The saved filenames follow the format: 
      `cell_user_counts_{window_duration}_windows_{cluster_method}_{cluster_label}.html`

    """
    trajectories_grouped = trajectories_df.groupby(clusters_col)

    for cluster_label, group_df in trajectories_grouped:

        # Get the counts of unique MAC addresses by H3 cell and time window
        users_per_h3_cell_time_window = utils.trajectories_to_h3_counts_gdf(group_df,
                                                                            count_type='users',
                                                                            window_duration=window_duration,
                                                                            sonar_type=sonar_type)
        
        # Extract the cluster method for the plot caption
        clusters_col_split =  clusters_col.split('_')[0:-1]
        cluster_method = '_'.join(clusters_col_split)

        # Convert to int for clarity when printing and saving (could be float)
        cluster_label = int(cluster_label)

        # Create the filename to save if needed
        if path_to_save:
            filename_to_save = os.path.join(path_to_save,f'cell_user_counts_{window_duration}_windows_{cluster_method}_{cluster_label}.html')
        else:
            raise Exception('`path_to_save` must be provided to save the plots')

        # Plot the counts timeline. Maps are not returned as there would be problems when visualizing them within a notebook
        utils.plot_counts_per_cell_timeline(sonar_type=sonar_type, 
                                            h3_cells_counts=users_per_h3_cell_time_window, 
                                            caption=f'Unique MAC addresses per H3 cell. {cluster_method}_{cluster_label}', 
                                            window_duration=window_duration, 
                                            filename_to_save=filename_to_save, 
                                            return_map=False)
        



def summarize_grouped_stats(trajectory_general_measures_df, clusters_col, columns_interest=None, stats=None, decimals=2):
    """
    Groups the DataFrame by a specified column and calculates summary statistics 
    for the specified columns. The count is displayed only once for each group.

    Parameters:
    - trajectory_general_measures_df (pd.DataFrame): The input DataFrame.
    - clusters_col (str): The column name representing cluster labels in `trajectory_general_measures_df`.
    - columns_interest (list, optional): List of columns to summarize. If None, all numerical columns are used.
    - stats (list, optional): List of statistics to include (default: ['mean', 'std', 'min', 'max']).
    - decimals (int, optional): Number of decimal places to round the results to (default: 2).
    
    Returns:
    - pd.DataFrame: Summary statistics table with count displayed only once per group.
    """
    if columns_interest is None:
        columns_interest = trajectory_general_measures_df.select_dtypes(include='number').columns.tolist()
    
    if stats is None:
        stats = ['mean', 'std', 'min', 'max']  # Default main statistics
    
    # Group and compute selected statistics
    grouped = trajectory_general_measures_df.groupby(clusters_col)[columns_interest].agg(stats)
    
    
    # Calculate count and proportion separately
    count = trajectory_general_measures_df.groupby(clusters_col).size().rename('count')
    total_count = count.sum()
    proportion = (count / total_count).rename('proportion')
    
    # Concatenate count at the top and transpose
    final_result = pd.concat([count, proportion, grouped], axis=1).T

    return final_result.round(decimals)




def plot_vendor_distribution_heatmap(df, cluster_column):
    """
    Plots a heatmap of the distribution of categories in 'group_column' by clusters in 'cluster_column'.
    Replaces None or NaN values in the group_column with 'Unknown' and sorts vendors by frequency.
    
    Parameters:
    - df: DataFrame containing the data
    - cluster_column: Column for cluster labels 
    """
    # Replace None or NaN values with 'Unknown'
    df['vendor_name'] = df['vendor_name'].fillna('Unknown')

    # Calculate the frequency of each vendor (count of appearances)
    vendor_frequencies = df['vendor_name'].value_counts()

    # Create a pivot table of group counts by cluster label
    vendor_cluster_matrix = pd.crosstab(df['vendor_name'], df[cluster_column])

    # Sort the rows by vendor frequency (most frequent first)
    vendor_cluster_matrix = vendor_cluster_matrix.loc[vendor_frequencies.index]

    # Normalize across columns (i.e., each column sums to 1)
    vendor_cluster_matrix_normalized = vendor_cluster_matrix.div(vendor_cluster_matrix.sum(axis=0), axis=1)

    # Plot heatmap with row-wise normalization
    plt.figure(figsize=(10, 4))
    sns.heatmap(vendor_cluster_matrix_normalized, annot=True, cmap='plasma', fmt='.3f', cbar_kws={'label': 'Normalized Count'})
    
    # Move cluster labels to the top and flip them
    plt.title(f'Heatmap of vendor_name Distribution by {cluster_column}')
    plt.xlabel('Cluster Label')
    plt.ylabel(None)
    plt.gca().xaxis.set_label_position('top')
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.gca().xaxis.set_ticks_position('top')
    
    # Remove the ticks (but keep the labels)
    plt.gca().tick_params(axis='x', which='both', bottom=False, top=False)
    plt.gca().tick_params(axis='y', which='both', left=False, right=False)

    plt.tight_layout(pad=0)
    
    plt.show()




def plot_stage_proportion_heatmaps_by_cluster(trajectories_df, clusters_col, sonar_type='night', path_to_save=None):
    """
    Plots heatmaps of the proportion of people present at different stages over time, grouped by clusters, for each day or night of the event.

    Parameters:
    -----------
    trajectories_df : pandas.DataFrame
        The input DataFrame containing the following columns:
            - 'tid': Unique identifiers for days or nights.
            - 'stage_renamed': Stage names or labels.
            - 'time_window_15min': Time windows in 15-minute intervals.
            - 'uid': Unique identifiers for individuals (e.g., attendees or staff).
            - Additional columns corresponding to clustering labels.

    clusters_col : str
        The name of the column in the DataFrame that contains cluster labels for grouping trajectories.

    sonar_type : str, optional, default='night'
        A label indicating the type of event (e.g., 'night'). This is included in the plot titles.

    path_to_save : str, optional, default=None
        The path to the directory where the heatmaps will be saved as PDFs. If None, the plots will not be saved.

    Returns:
    --------
    None
        Displays heatmaps for each cluster and optionally saves them to the specified path. Each heatmap shows the proportion of people across stages for the corresponding cluster over time.
    """
    trajectories_grouped = trajectories_df.groupby(clusters_col)

    for cluster_label, group_df in trajectories_grouped:
        people_counts_window_stage_heatmap = utils.proportion_people_stage(group_df)

        # Extract the cluster method for displaying and saving
        clusters_col_split =  clusters_col.split('_')[0:-1]
        cluster_method = '_'.join(clusters_col_split)

        # Convert to int for clarity when printing and saving (could be float)
        cluster_label = int(cluster_label)

        # Create the filename to save if needed
        if path_to_save:
            filename_to_save = os.path.join(path_to_save,f'people_counts_15min_heatmap{cluster_method}_{cluster_label}.pdf')
        else:
            filename_to_save = None
        
        print(f'People count time window heatmaps for {cluster_method}_{cluster_label}')
        utils.plot_stage_proportion_heatmaps(data=people_counts_window_stage_heatmap,
                                            filename_to_save=filename_to_save)





#--------------------------------------------------------------------------



def get_medoids_indices(X, labels, exclude_noise=True):
    """
    Computes the medoids indices for clusters based on the provided data and labels.

    Parameters:
        X (array-like): The dataset containing all data points, where each row represents a data point (e.g., a 2D array).
        labels (array-like): Cluster labels assigned to each data point. Noise points should be labeled as -1.
        exclude_noise (bool, optional): Whether to exclude noise points (label -1) from the computation. Defaults to True.

    Returns:
        medoid_indices (ndarray): A NumPy array containing the indices of the medoids in the original dataset.

    Notes:
        - Medoids are computed as the points within each cluster that minimize the sum of pairwise distances 
          to all other points in the same cluster.
        - If `exclude_noise` is True, points labeled as noise (-1) are ignored during medoid calculation.
    """
    if exclude_noise:
        labels=labels[labels != -1]

    unique_clusters = np.unique(labels)
    medoid_indices = []

    for cluster in unique_clusters:
        # Extract cluster indices and points
        cluster_indices = np.where(labels == cluster)[0]
        cluster_points = X[cluster_indices]
        
        # Compute pairwise distances within the cluster
        distance_matrix = pairwise_distances(cluster_points, cluster_points, metric='euclidean')

        # Find the medoid (point with the smallest sum of distances)
        medoid_idx_local = np.argmin(distance_matrix.sum(axis=1))
        medoid_idx_global = cluster_indices[medoid_idx_local]  # Convert to global index
        
        # Store medoid index
        medoid_indices.append(medoid_idx_global)  

    return np.array(medoid_indices)




def plot_traj_medoid_clusters(path_to_save, clusters_col, medoid_indices,
                              trajectory_general_measures_df, trajectories_events_df, sequence_stops_df,
                              show_maps=False, plot_stops=True):
    trajectory_general_grouped = trajectory_general_measures_df.groupby(clusters_col)
    for cluster_label, group in trajectory_general_grouped:
        medoid_uid_tid = group.loc[group[medoid_indices]==True, ['uid','tid']].to_numpy().flatten()
        medoid_uid = medoid_uid_tid[0]
        medoid_tid = medoid_uid_tid[1]

        full_trajectory = trajectories_events_df.loc[(trajectories_events_df['uid']==medoid_uid) &
                                                     (trajectories_events_df['tid']==medoid_tid)]
        
        sequence_stops = sequence_stops_df.loc[(sequence_stops_df['uid']==medoid_uid) &
                                               (sequence_stops_df['tid']==medoid_tid)]
        

        # Extract the cluster method for the plot caption
        clusters_col_split =  clusters_col.split('_')[0:-1]
        cluster_method = '_'.join(clusters_col_split)
        
        filename_to_save = os.path.join(path_to_save,f'traj_{cluster_method}_{cluster_label}_medoid.html')
        visualize_trajectory_and_stops(full_trajectories_df=full_trajectory, sequence_stops_df=sequence_stops,
                                       plot_stops=plot_stops,
                                       show_map=show_maps,
                                       filename_to_save=filename_to_save)



def plot_exemplar_cluster(hdbscan_experiment, parameters_experiment,
                          trajectory_general_measures_df, trajectories_events_df, sequence_stops_df, 
                          cluster_id=0, exemplar_idx=0, filename_to_save=None, 
                          show_maps=False, plot_stops=True):

    print(f'Alternative for cluster {cluster_id}:')
    exemplar_index = extract_exemplar_index(hdbscan_experiment=hdbscan_experiment,
                                            parameters_experiment=parameters_experiment,
                                            cluster_id=cluster_id, exemplar_idx=exemplar_idx) 
    exemplar_uid_tid = trajectory_general_measures_df.iloc[exemplar_index].to_numpy().flatten()
    exemplar_uid = exemplar_uid_tid[0]
    exemplar_tid = exemplar_uid_tid[1]

    full_trajectory = trajectories_events_df.loc[(trajectories_events_df['uid']==exemplar_uid) &
                                                 (trajectories_events_df['tid']==exemplar_tid)]

    sequence_stops = sequence_stops_df.loc[(sequence_stops_df['uid']==exemplar_uid) &
                                           (sequence_stops_df['tid']==exemplar_tid)]
    
    visualize_trajectory_and_stops(full_trajectories_df=full_trajectory, sequence_stops_df=sequence_stops,
                                   plot_stops=plot_stops,
                                   show_map=show_maps,
                                   filename_to_save=filename_to_save) 
  
        





def plot_branch_condensed_tree(branch_detector, idx=None):
    """
    Plots the condensed tree for a specific cluster branch in a branch detection analysis.

    Parameters:
        branch_detector: An object containing branch detection results, including 
                         branch persistences and condensed trees. Typically produced 
                         by clustering or hierarchical algorithms.
        idx (int, optional): The index of the cluster to plot. If None, the cluster 
                             with the most branches is selected automatically.
    """
    if not idx:
        idx = np.argmax([len(x) for x in branch_detector.branch_persistences_])
    
    branch_detector.cluster_condensed_trees_[idx].plot(select_clusters=True)

    plt.ylabel("Eccentricity")
    plt.title(f"Branches in cluster {idx}")
    plt.show()



def extract_exemplar_index(hdbscan_experiment, parameters_experiment, cluster_id=1, exemplar_idx=1):
    # Extract the full array of embeddings
    chosen_lambda = parameters_experiment[0]
    embeddings = hdbscan_experiment.multi_distances_embeddings[chosen_lambda]['embeddings'] # Shape (n_trajectories, d)

    # Extract the exemplar (one observation of the embeddings)
    hdbscan_instance = hdbscan_experiment.hdbscan_instances_dict[parameters_experiment]
    exemplar = hdbscan_instance.exemplars_[cluster_id][exemplar_idx]  # Shape (d,)

    # Check which row matches the observation
    matching_index = np.where(np.all(embeddings == exemplar, axis=1))[0]

    # Output the result
    if matching_index.size > 0:
        print(f"The exemplar selected for cluster {cluster_id} has index: {matching_index[0]}")
    else:
        raise Exception("No matching observation found.")

    return matching_index[0]





def visualize_trajectory_and_stops(full_trajectories_df, sequence_stops_df=None, plot_stops=False,
                                    example_idx=0, given_uid=None, background_sonar='night',
                                    show_map=True, filename_to_save=None):
    """
    Visualizes user trajectory and (optionally) stops on a map.

    Parameters:
        full_trajectories_df (DataFrame): DataFrame containing full trajectories. 
                                          Must include columns for longitude ('lng'), latitude ('lat'), 
                                          datetime ('datetime'), user ID ('uid'), and trajectory ID ('tid').
        sequence_stops_df (DataFrame, optional): DataFrame containing the sequence of stops. 
                                                 Must include columns for centroid longitude ('lng_centroid'), 
                                                 centroid latitude ('lat_centroid'), stop start time ('start_time_stop'), 
                                                 user ID ('uid'), and trajectory ID ('tid'). Defaults to None.
        plot_stops (bool, optional): Whether to overlay the sequence of stops on the trajectory. Defaults to False.
        example_idx (int, optional): Index to select a specific user from the dataframes if `given_uid` is not provided. Defaults to 0.
        given_uid (str, optional): User ID to visualize. If None, the user is selected using `example_idx`. Defaults to None.
        background_sonar (str, optional): Background map style for visualization (e.g., 'night', 'day'). Defaults to 'night'.
        show_map (bool, optional): Whether to display the generated map interactively. Defaults to True.
        filename_to_save (str, optional): If provided, the map will be saved to this file. Defaults to None.

    Returns:
        None: The function displays the map interactively or saves it to a file, but does not return any value.
    """
    # Select a single user
    if given_uid is None:
        example_uid = sequence_stops_df['uid'].unique()[example_idx]
    else:
        example_uid = given_uid

    full_trajectories_example_uid = full_trajectories_df[full_trajectories_df['uid']==example_uid]   
    tdf_full_example_id = skmob.TrajDataFrame(full_trajectories_example_uid,
                                     longitude='lng', latitude='lat',
                                     datetime='datetime',
                                     user_id='uid',
                                     trajectory_id='tid')

    
    # Plot the full trajectory
    tdf_full_map = utils.plot_trajectories_in_context(tdf=tdf_full_example_id,
                                                        background=background_sonar, max_users=1)

    # Plot the sequence of stops on top of the trajetory if needed
    if plot_stops and sequence_stops_df is not None:
        
        sequence_events_example_uid = sequence_stops_df[sequence_stops_df['uid']==example_uid]
        tdf_sequence_events_example_id = skmob.TrajDataFrame(sequence_events_example_uid,
                                                             longitude='lng_centroid', latitude='lat_centroid',
                                                             datetime='start_time_stop',
                                                             user_id='uid',
                                                             trajectory_id='tid')

        tdf_full_map = utils.plot_trajectories_in_context(tdf=tdf_sequence_events_example_id,
                                                          background=tdf_full_map, max_users=1,
                                                          dashArray='5,6',
                                                          weight=3, hex_color='black', opacity=0.5,
                                                          start_end_markers=False)
    # Save to file if requested
    if filename_to_save:
        tdf_full_map.save(filename_to_save)
        print(f'File saved in {filename_to_save}.')

    # Return the map object
    if show_map:
        display(tdf_full_map)



#--------------------------------------------------------------------------------------
def plot_parallel_coordinates(df, cluster_col, feature_columns, feature_labels=None, offset_colormap=0, normalized=True, title_fontsize=14, label_fontsize=14, tick_fontsize=13, legend_fontsize=10):
    """
    Plot a parallel coordinates plot for the means of the given features across clusters.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the data.
    - cluster_col (str): The column name containing cluster labels.
    - feature_columns (list of str): A list of column names to be used in the plot.
    - feature_labels (list of str, optional): Custom labels for the features.
    - offset_colormap (int, optional): An offset value for shifting colormap indices to adjust cluster colors (default is 0).
    - normalized (bool): Whether to normalize the feature columns (default is True).
    - title_fontsize (int): Font size for the plot title.
    - label_fontsize (int): Font size for axis labels.
    - tick_fontsize (int): Font size for tick labels.
    - legend_fontsize (int): Font size for legend.
    """
    # Remove noise cluster (-1)
    df = df[df[cluster_col] != -1].copy()

    # Normalize the feature columns if required
    if normalized:
        scaler = StandardScaler()
        df[feature_columns] = scaler.fit_transform(df[feature_columns])

    # Compute the mean for each feature within each cluster
    cluster_means = df.groupby(cluster_col)[feature_columns].mean()

    # Create a figure and axis for the plot
    fig, ax = plt.subplots(figsize=(11, 4))

    # Get cluster labels
    cluster_labels = cluster_means.index

    # Turn on vertical grid
    ax.grid(True, axis='x', alpha=0.8, linewidth=2.5)

    # Use custom feature labels if provided, otherwise use column names
    if feature_labels is None:
        feature_labels = feature_columns

    # Plot parallel coordinates for each cluster's mean values
    for i, cluster in enumerate(cluster_labels):
        cluster_data = cluster_means.loc[cluster]
        ax.plot(cluster_data, color=plt.cm.tab10(i + offset_colormap), alpha=0.7, marker='o', label=f'Cluster {cluster}')  # Shift colormap by +1 o account for noise removal

    # Set labels and title
    ax.set_title('Parallel Coordinates Plot for Clusters (Means)', fontsize=title_fontsize)
    ax.set_xticks(range(len(feature_columns)))
    ax.set_xticklabels(feature_labels, fontsize=tick_fontsize)
    ax.set_ylabel('Z-Normalized Feature Value', fontsize=label_fontsize)
    ax.tick_params(axis='y', labelsize=tick_fontsize)

    # Add a legend
    ax.legend(loc='best', fontsize=legend_fontsize)

    # Display the plot
    plt.tight_layout()
    plt.show()