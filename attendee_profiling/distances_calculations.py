import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.stats import norm
import seaborn as sns
import umap
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
import tslearn
from kneed import KneeLocator
from tslearn.metrics import lcss
from tslearn.metrics import dtw


import constants

# General settings for plotting
sns.set_theme('paper', style="dark")

plt.rcParams.update({
    "axes.labelsize": 12,
    'legend.fontsize': 11,
    'axes.titlesize': 13
})

def compute_combined_distance(distances_1, distances_2, lambda_value):
    """
    Compute a weighted sum of two distance matrices.

    Parameters:
    distances_1 (numpy.ndarray): The first distance matrix.
    distances_2 (numpy.ndarray): The second distance matrix.
    lambda_value (float): The weight applied to the second distance matrix.

    Returns:
    numpy.ndarray: The combined distance matrix.
    """
    return distances_1 + lambda_value * distances_2


#--------------------------------------------------------------------------


def general_features_df_to_normalized_numpy(trajectory_general_features_df):
    """
    Convert a DataFrame of general features to a normalized NumPy array.
    Each feature is normalized with respect to the values of that feature across 
    all trajectories (z-score normalization with StandardScaler).

    Parameters:
    trajectory_general_features_df (pandas.DataFrame): 
        A DataFrame containing general features, indexed by 'uid' and 'tid'.

    Returns:
    numpy.ndarray:
        A normalized NumPy array of general features.
    """
    # Reindex the Dataframe to leave the columns for features only
    reindexed_df = trajectory_general_features_df.set_index(['uid','tid'])

    # Convert to numpy array for computing the distances    
    general_features_array = reindexed_df.to_numpy()

    # Normalize the general features 
    # Each feature normalized with respect to the values of that feature across all trajectories
    scaler = StandardScaler()
    general_features_array = scaler.fit_transform(general_features_array)

    return general_features_array




def compute_general_euclidean_distance(trajectory_general_features_df):
    """
    Compute pairwise Euclidean distances from a DataFrame of general trajectory features.

    Parameters:
    trajectory_general_features_df (pandas.DataFrame): DataFrame containing trajectory general features, indexed by 'uid' and 'tid'.

    Returns:
    numpy.ndarray: Pairwise Euclidean distance matrix.
    """
    # Convert the DataFrame to a normalized NumPy array
    general_features_array_normalized = general_features_df_to_normalized_numpy(trajectory_general_features_df)

    # Compute pairwise Euclidean distances
    distance_matrix = pairwise_distances(general_features_array_normalized, metric='euclidean')

    return distance_matrix



def compute_general_euclidean_distance_pca(trajectory_general_features_df, n_components):
    """
    Normalizes the general features in the input DataFrame, reduces their dimensionality using PCA,
    and calculates pairwise Euclidean distances between the reduced feature vectors.

    Parameters:
    trajectory_general_features_df (pandas.DataFrame):
        A DataFrame containing general features, indexed by 'uid' and 'tid'.
    n_components (int):
        The number of principal components to retain for PCA dimensionality reduction.

    Returns:
    numpy.ndarray:
        A pairwise Euclidean distance matrix computed from the normalized general features.
    """
    # Convert the DataFrame to a normalized NumPy array
    general_features_array_normalized = general_features_df_to_normalized_numpy(trajectory_general_features_df)

    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(general_features_array_normalized)

    # Compute pairwise Euclidean distances
    distance_matrix = pairwise_distances(general_features_array_normalized, metric='euclidean')

    return distance_matrix





def _tw_features_df_to_normalized_numpy(tw_features_df):
    """
    Convert a time window features DataFrame to a normalized NumPy array and return the shape.
    This function groups the DataFrame by 'uid' and 'tid' to create a 3D structure with shape
    (n_trajectories, T, d), flattens it into a 2D array, and normalizes the features using z-score 
    normalization (StandardScaler). Each feature at a specific time window is normalized with 
    respect to the values of that feature across all trajectories.

    Parameters:
    tw_features_df (pandas.DataFrame): DataFrame containing time window features, indexed by 'uid' and 'tid'.

    Returns:
    tuple:
        - numpy.ndarray: Normalized NumPy array of shape (n_trajectories, T * d), where n_trajectories is the number of trajectories,
          T is the number of time windows, and d is the number of features.
        - tuple: A tuple containing the original shape (n_trajectories, T, d).
    """
    # Pivot the DataFrame to create a 3D-like structure (n trajectories x T time windows x d features)
    # Group by (uid, tid) to aggregate rows for each trajectory
    trajectory_groups = tw_features_df.groupby(['uid', 'tid'])
    
    # Create a feature array with shape (n, T, d)
    # n = number of trajectories, T = number of time windows, d = number of features
    feature_array = np.stack([group.iloc[:, 4:].to_numpy() for _, group in trajectory_groups])
    
    # Flatten the feature array for each trajectory into a single vector
    n_traj, T, d = feature_array.shape
    flattened_features = feature_array.reshape(n_traj, T * d)  # Shape: (n, T*d)

    # Normalize the flattened features 
    # Each feature at a specific time window is normalized with respect to the values of that feature across all trajectories
    scaler = StandardScaler()
    flattened_features = scaler.fit_transform(flattened_features)

    return flattened_features, (n_traj, T, d)
    



def compute_ts_euclidean_distance(tw_features_df):
    """
    Compute pairwise Euclidean distances from a DataFrame of time window features.

    Parameters:
    tw_features_df (pandas.DataFrame): DataFrame containing time window features, indexed by 'uid' and 'tid'.

    Returns:
    numpy.ndarray: Pairwise Euclidean distance matrix.
    """
    # Convert the DataFrame to a normalized NumPy array and get the original shape
    flattened_features_normalized, original_shape = _tw_features_df_to_normalized_numpy(tw_features_df)
    
    # Extract the shape components
    n_traj, T, d = original_shape
    
    # Compute pairwise Euclidean distances
    distance_matrix = pairwise_distances(flattened_features_normalized, metric='euclidean')
    
    # Normalize distances by T (all trajectories have the same number of time windows)
    distance_matrix = distance_matrix / T
    
    return distance_matrix




def compute_ts_euclidean_distance_pca(tw_features_df, n_components):
    """
    Compute pairwise Euclidean distances from a DataFrame of time window features after reducing dimensionality with PCA.

    Parameters:
    tw_features_df (pandas.DataFrame): DataFrame containing time window features, indexed by 'uid' and 'tid'.
    n_components (int): The number of principal components for PCA.

    Returns:
    numpy.ndarray: Pairwise Euclidean distance matrix in the PCA-transformed space.
    """
    # Convert the DataFrame to a normalized NumPy array and get the original shape
    flattened_features_normalized, original_shape = _tw_features_df_to_normalized_numpy(tw_features_df)
    
    # Extract the shape components
    n_traj, T, d = original_shape

    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(flattened_features_normalized)

    # Compute pairwise Euclidean distances in the PCA-transformed space
    distance_matrix = pairwise_distances(reduced_features, metric='euclidean')

    # Normalize distances by T (since all trajectories have the same number of time windows)
    distance_matrix = distance_matrix / T

    return distance_matrix




def dtw_lcss_computation(settings, trajectories_array, verbose=False):
    """
    Compute pairwise distances between trajectories using DTW or LCSS and return a distance matrix.

    Parameters:
    settings (dict): Configuration dictionary containing 'distance_type' ('DTW' or 'LCSS') and 'epsilon_LCSS' (for LCSS).
    trajectories_array (numpy.ndarray): Array of trajectories with shape (n_trajectories, max_length, n_features).
    verbose (bool, optional): If True, print progress messages (default is False).

    Returns:
    numpy.ndarray: A symmetric distance matrix of shape (n_trajectories, n_trajectories) containing pairwise distances.
    """
    # Number of trajectories
    n_trajectories = trajectories_array.shape[0]

    # Initialize the distance matrix (n_trajectories x n_trajectories)
    distance_matrix = np.zeros((n_trajectories, n_trajectories))

    # Fill the distance matrix with distances
    for i in range(n_trajectories):
        for j in range(i + 1, n_trajectories):

            # Trajectories may have different lengths, so it is needed to 
            # mask the them to pass only valid arrays for the distance functions (they don't handle nan values)
            mask_i = ~np.isnan(trajectories_array[i,:,0])
            mask_j = ~np.isnan(trajectories_array[j,:,0])

            masked_trajectory_i = trajectories_array[i,mask_i,:]
            masked_trajectory_j = trajectories_array[j,mask_j,:]

            # Actual distance computation
            if settings['distance_type'] == 'LCSS':
                distance = 1 - tslearn.metrics.lcss(masked_trajectory_i, masked_trajectory_j, eps=settings['epsilon_LCSS'])
            elif settings['distance_type'] == 'DTW':
                distance = tslearn.metrics.dtw(masked_trajectory_i, masked_trajectory_j)
                
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance  # Distances are symmetric in this case


        if verbose:
            print(f'Distances computed for trajectory {i}')
    
    return distance_matrix



#--------------------------------------------------------------------------




def _check_pca_variance(features_array, variance_threshold=0.95):
    """
    Apply PCA to a feature array, compute the cumulative explained variance, and find the number of components
    needed to reach the given variance threshold and the knee point.

    Parameters:
    features_array (numpy.ndarray): The array of features to be analyzed.
    variance_threshold (float, optional): The cumulative variance threshold (default is 0.95).

    Returns:
    int: The number of components needed to reach the variance threshold.
    int: The number of components corresponding to the knee point.
    numpy.ndarray: Cumulative explained variance for each number of components.
    """
    # Apply PCA
    pca = PCA()
    pca.fit(features_array)

    # Compute cumulative explained variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    # Find the number of components needed for the given variance threshold
    n_components_threshold = np.searchsorted(cumulative_variance, variance_threshold) + 1

    # Find the knee (optimal number of components)
    knee_locator = KneeLocator(
        x=np.arange(1, len(cumulative_variance) + 1),
        y=cumulative_variance,
        curve='concave',
        direction='increasing'
    )
    knee_components = knee_locator.knee

    # Plot variance explained

    # Changing the plotting settings locally
    plt.rcParams.update({
    "axes.labelsize": 14,
    'legend.fontsize': 12.5,
    'axes.titlesize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11
    })

    plt.figure(figsize=(7.5, 5))
    plt.plot(np.arange(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='-')
    plt.axhline(y=variance_threshold, color='r', linestyle='--', label=f"{variance_threshold*100:.0f}% variance threshold")
    plt.axvline(x=n_components_threshold, color='g', linestyle='--', label=f"{n_components_threshold} components ({variance_threshold*100:.0f}% var.)")
    
    if knee_components is not None:
        variance_at_knee = cumulative_variance[knee_components - 1]
        plt.axhline(y=variance_at_knee, color='tab:purple', linestyle='--', label=f"{variance_at_knee*100:.0f}% variance (knee)")
        plt.axvline(x=knee_components, color='b', linestyle='--', label=f"Knee at {knee_components} components")

    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Explained Variance")
    plt.legend(loc="upper left")
    plt.grid()
    plt.show()

    print(f"Number of components for {variance_threshold*100:.0f}% variance: {n_components_threshold}")
    print(f"Knee detected at: {knee_components} components")

    return n_components_threshold, knee_components, cumulative_variance




def check_tw_pca_variance(tw_features_df, variance_threshold=0.95):
    """
    Compute the number of components needed to reach a specified variance threshold using PCA
    for a DataFrame of time window features, and plot the cumulative explained variance.

    Parameters:
    tw_features_df (pandas.DataFrame): DataFrame containing time window features, containing 'uid' and 'tid'.
    variance_threshold (float, optional): The cumulative variance threshold (default is 0.95).

    Returns:
    int: The number of components needed to reach the variance threshold.
    int: The number of components corresponding to the knee point.
    numpy.ndarray: Cumulative explained variance for each number of components.
    """

    # Convert the DataFrame to a normalized NumPy array and discard the original shape
    flattened_features_normalized, _ = _tw_features_df_to_normalized_numpy(tw_features_df)

    # Compute the number of components needed for the given variance threshold using PCA
    n_components_threshold, knee_components, cumulative_variance = _check_pca_variance(features_array=flattened_features_normalized, 
                                                                                       variance_threshold=variance_threshold)
    
    return n_components_threshold, knee_components, cumulative_variance





def check_general_pca_variance(trajectory_general_features_df, variance_threshold=0.95):
    """
    Compute the number of components needed to reach a specified variance threshold using PCA
    for a DataFrame of general features, and plot the cumulative explained variance.

    Parameters:
    trajectory_general_features_df (pandas.DataFrame): DataFrame containing general features, 
    containing 'uid' and 'tid'.
    variance_threshold (float, optional): The cumulative variance threshold (default is 0.95).

    Returns:
    int: The number of components needed to reach the variance threshold.
    int: The number of components corresponding to the knee point.
    numpy.ndarray: Cumulative explained variance for each number of components.
    """

    # Convert the DataFrame to a normalized NumPy array
    general_features_array_normalized = general_features_df_to_normalized_numpy(trajectory_general_features_df)

    # Compute the number of components needed for the given variance threshold using PCA
    n_components_threshold, knee_components, cumulative_variance = _check_pca_variance(features_array=general_features_array_normalized, 
                                                                                       variance_threshold=variance_threshold)
    
    return n_components_threshold, knee_components, cumulative_variance





#--------------------------------------------------------------------------

def normalize_min_max(distance_matrix):
    """
    Function to normalize a distance matrix using min-max scaling
    """
    min_val = distance_matrix.min()
    max_val = distance_matrix.max()
    return (distance_matrix - min_val) / (max_val - min_val)



def normalize_z_score(distance_matrix):
    """
    Function to normalize a distance matrix using z-score normalization
    """
    mean_val = distance_matrix.mean()
    std_val = distance_matrix.std()
    return (distance_matrix - mean_val) / std_val



def normalize_by_max(distance_matrix):
    """
    Function to normalize a distance matrix by the maximum absolute value
    """
    max_val = distance_matrix.max()
    return distance_matrix / max_val



def normalize_by_mean(distance_matrix):
    """
    Function to normalize a distance matrix by the mean
    """
    mean_val = distance_matrix.mean()
    return distance_matrix / mean_val



def normalize_by_std(distance_matrix):
    """
    Function to normalize a distance matrix by the standard deviation
    """
    std_val = distance_matrix.std()
    return distance_matrix / std_val


#--------------------------------------------------------------------------

def check_distance_distribution(distance_matrix):
    """
    Analyze and plot the distribution of pairwise distances from a distance matrix.

    Parameters:
    distance_matrix (numpy.ndarray): A square matrix containing pairwise distances.

    Returns:
    None
    """
    # Flatten the distance matrix and remove diagonal entries
    distances = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]

    # Fit a normal distribution to the data
    mu, sigma = norm.fit(distances)  # Mean and standard deviation of the fitted normal distribution

    # Plot the histogram of the data
    plt.hist(distances, bins=50, density=True, alpha=0.6, color="skyblue", edgecolor="black", label="Pairwise Distances")

    # Plot the fitted normal distribution
    x = np.linspace(min(distances), max(distances), 500)
    fitted_pdf = norm.pdf(x, mu, sigma)
    plt.plot(x, fitted_pdf, "r-", lw=2, label=f"Fitted Normal (μ={mu:.2f}, σ={sigma:.2f})")

    # Add labels and legend
    plt.xlabel("Distance")
    plt.ylabel("Density")
    plt.title("Distribution of Pairwise Distances with Fitted Normal Distribution")
    plt.legend()

    # Show the plot
    plt.show()


def plot_hierarchical_clustering_heatmap(distance_matrix, label_matrix=None, method='average', cmap='viridis', heatmap_params={}):
    """
    Perform hierarchical clustering on a distance matrix and plot the heatmap.

    Parameters:
    - distance_matrix: 2D array-like, the distance matrix to be clustered.
    - method: str, the linkage method to be used for hierarchical clustering. Default is 'average'.
    - cmap: str, the colormap to be used for the heatmap. Default is 'viridis'.
    """
    # Perform hierarchical clustering
    linkage_matrix = linkage(distance_matrix, method=method)
    
    # Get the new ordering of indices from the dendrogram
    ordered_indices = leaves_list(linkage_matrix)
    
    # Reorder the distance matrix
    ordered_matrix = distance_matrix[ordered_indices, :][:, ordered_indices]
    
    # Plot heatmap
    ax = sns.heatmap(ordered_matrix, cmap=cmap,
                     **heatmap_params
                     )
        
    # Set colorbar label size directly
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=18)

    
    # Set title
    title_text = "Reordered Distance Matrix (Hierarchical Clustering)"
    if label_matrix:
        title_text = f"Reordered Distance Matrix (Hierarchical Clustering) - {label_matrix}"
        plt.title(title_text, fontsize=12)
    
    plt.show()


#--------------------------------------------------------------------------



def compute_umap_embeddings(distances_dict, umap_components=2, umap_params={}, save_path=None): 
    """
    Compute UMAP embeddings from a dictionary of distance matrices and optionally save the results.

    Parameters:
    distances_dict (dict): A dictionary where keys are lambda values and values are distance matrices.
    umap_components (int, optional): Number of dimensions for UMAP embeddings (default is 2).
    umap_params (dict, optional): Additional parameters for UMAP initialization (default is {}).
    save_path (str, optional): Path to save the computed embeddings as a pickle file (default is None).

    Returns:
    dict: A dictionary with lambda values as keys and dictionaries containing distances and embeddings as values.
    """
    distances_and_embeddings_dict = {}
    for lambda_val, distances in distances_dict.items():
        embeddings_dict = {}
        
        umap_instance = umap.UMAP(metric='precomputed', n_components=umap_components,
                                  random_state=constants.RANDOM_SEED,
                                  **umap_params)
        umap_embedding = umap_instance.fit_transform(distances)
        
        distances_and_embeddings_dict[lambda_val] = {'distances':distances, 'embeddings':umap_embedding}
        print(f'Computed the embeddings for lambda = {lambda_val}')

    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(distances_and_embeddings_dict, f)
        print(f'Saved UMAP embeddings to {save_path}')
    
    return distances_and_embeddings_dict




def plot_embeddings(embedding, labels=None, ax=None, title=None, dim=2, medoid_indices=None):
    """
    Plots a single 2D or 3D embedding on a given axis or in a new figure.

    Parameters:
        embedding (array-like): The transformed data (2D or 3D coordinates).
        labels (array-like): Optional, the labels for each data point.
        ax (matplotlib.axes.Axes): Optional, the axis to plot on. If None, a new figure and axis will be created.
        title (str): Optional, the title for the plot.
        dim (int): The dimension of the plot (2 for 2D, 3 for 3D).
        medoid_indices (array-like): Indices of medoid points to highlight.
    """
    # Map the labels to their categorical colors (if provided)
    unique_labels = np.unique(labels)

    if -1 in unique_labels:
        bounds = np.arange(len(unique_labels) + 1) - 1.5
    else:
        bounds = np.arange(len(unique_labels) + 1) - 0.5

    norm = mcolors.BoundaryNorm(boundaries=bounds, 
                                ncolors=len(unique_labels))

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d' if dim == 3 else None)

    # Define colormap and assign noise (-1) to a specific color
    if len(unique_labels) <= 10:
        cmap = mcolors.ListedColormap([plt.cm.tab10(i) for i, label in enumerate(unique_labels)])
    else:
        cmap = mcolors.ListedColormap([plt.cm.tab20(i) for i, label in enumerate(unique_labels)])

    if dim == 2:
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1],
                             c=labels, cmap=cmap, norm=norm, s=50, edgecolor='k', alpha=0.7)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.grid(True)

    elif dim == 3:
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], 
                             c=labels, cmap=cmap, norm=norm, s=50, edgecolor='k', alpha=0.7)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.set_zlabel("Component 3")

    # Highlight medoids
    if medoid_indices is not None and len(medoid_indices) > 0:
        ax.scatter(
            embedding[medoid_indices, 0], embedding[medoid_indices, 1],
            *(embedding[medoid_indices, 2],) if dim == 3 else (),
            c='black', edgecolors='white', s=100, marker='X', label="Medoids"
        )

    if title:
        ax.set_title(title)

    # Fix colorbar to show only discrete labels correctly
    if labels is not None:
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, 
                            boundaries=bounds, 
                            spacing='proportional', 
                            ticks=unique_labels)
        cbar.ax.tick_params(size=0)  # Remove tick marks

        cbar.outline.set_visible(False)

    if ax is None:
        plt.show()




def plot_multiple_embeddings(multi_distances_embeddings, cols=3, dim=2):
    """
    Plots a grid of 2D or 3D embeddings.

    Parameters:
        embeddings_dict (dict): Dictionary with lambda values as keys and embeddings as values.
        cols (int): Number of columns in the plot grid.
        dim (int): The dimension of the plots (2 for 2D, 3 for 3D).
    """
    num_plots = len(multi_distances_embeddings)
    rows = (num_plots // cols) + (num_plots % cols > 0)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows),
                             subplot_kw={'projection': '3d'} if dim == 3 else {})

    # Use the plot_mds_embedding function for each ax
    for idx, ((lambda_val, distances_embeddings), ax) in enumerate(zip(multi_distances_embeddings.items(), axes.flatten())):
        embeddings = distances_embeddings['embeddings']
        title = f'Lambda = {lambda_val}'
        plot_embeddings(embeddings, ax=ax, title=title, dim=dim)

    # Hide any unused subplots
    for i in range(idx + 1, rows*cols):
        fig.delaxes(axes.flat[i])

    plt.tight_layout()
    plt.show()



def check_multi_n_neighbors_umap(n_neighbors_list, combined_distances_dict, 
                                 path_rw, distance_type,
                                 read_write='read'):
    """
    Compute or load UMAP embeddings for multiple n_neighbors values and plot the results.

    Parameters:
    n_neighbors_list (list): List of n_neighbors values to iterate over.
    combined_distances_dict (dict): Dictionary of combined distance matrices.
    path_rw (str): Path for reading or writing embeddings.
    read_write (str, optional): Mode to either 'read' or 'write' embeddings (default is 'read').

    Returns:
    None
    """
    for n_neighbors in n_neighbors_list:
        embeddings_filename = os.path.join(path_rw,f'umaps_{distance_type}_2d_{n_neighbors}neighbors.pkl')
        if read_write=='write':
            embeddings = compute_umap_embeddings(distances_dict=combined_distances_dict,
                                                 umap_components=2,
                                                 umap_params={'min_dist':0, 'n_neighbors':n_neighbors},
                                                 save_path=embeddings_filename)
        elif read_write=='read':
            with open(embeddings_filename, 'rb') as f:
                embeddings = pickle.load(f)
                print(f'Loaded UMAP distance embeddings from {embeddings_filename}')

        print(f'\nPlot for n_neighbors = {n_neighbors}')
        plot_multiple_embeddings(multi_distances_embeddings=embeddings, cols=3)



#--------------------------------------------------------------------------



def check_svd_variance(distance_matrix, variance_threshold=0.99):
    """
    Compute the number of components needed to reach a specified variance threshold using Truncated SVD,
    and plot the cumulative explained variance.

    Parameters:
    distance_matrix (numpy.ndarray): The distance matrix to be analyzed.
    variance_threshold (float, optional): The cumulative variance threshold (default is 0.99).

    Returns:
    int: The number of components needed to reach the variance threshold.
    int: The optimal number of components (knee point).
    numpy.ndarray: Cumulative explained variance for each number of components.
    """
    # Normalize the flattened features
    scaler = StandardScaler()
    distance_matrix_scaled = scaler.fit_transform(distance_matrix)

    # Apply Truncated SVD
    svd = TruncatedSVD(n_components=min(distance_matrix_scaled.shape)-1)
    svd.fit(distance_matrix_scaled)

    # Compute cumulative explained variance
    cumulative_variance = np.cumsum(svd.explained_variance_ratio_)

    # Find the number of components needed for the given variance threshold
    n_components_threshold = np.searchsorted(cumulative_variance, variance_threshold) + 1

    # Find the knee (optimal number of components)
    knee_locator = KneeLocator(
        x=np.arange(1, len(cumulative_variance) + 1),
        y=cumulative_variance,
        curve='concave',
        direction='increasing'
    )
    knee_components = knee_locator.knee

    # Plot variance explained
    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='-')
    plt.axhline(y=variance_threshold, color='r', linestyle='--', label=f"{variance_threshold*100:.0f}% variance threshold")
    plt.axvline(x=n_components_threshold, color='g', linestyle='--', label=f"{n_components_threshold} components (threshold)")

    if knee_components is not None:
        variance_at_knee = cumulative_variance[knee_components - 1]
        plt.axhline(y=variance_at_knee, color='tab:purple', linestyle='--', label=f"{variance_at_knee*100:.0f}% variance threshold")
        plt.axvline(x=knee_components, color='b', linestyle='--', label=f"Knee at {knee_components} components")

    plt.xlabel("Number of Singular Values")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("SVD Explained Variance")
    plt.legend()
    plt.grid()
    plt.show()

    print(f"Number of components for {variance_threshold*100:.0f}% variance: {n_components_threshold}")
    print(f"Knee detected at: {knee_components} components")

    return n_components_threshold, knee_components, cumulative_variance



def print_explained_variance_svd(distance_matrix, n_components):
    """
    Apply Truncated SVD to reduce the dimensionality of a distance matrix and compute the explained variance.

    Parameters:
    distance_matrix (numpy.ndarray): The distance matrix to be reduced.
    n_components (int): The number of components for Truncated SVD.

    Returns:
    numpy.ndarray: The reduced matrix.
    float: The explained variance.
    """
    # Standardize the matrix
    scaler = StandardScaler()
    distance_matrix_scaled = scaler.fit_transform(distance_matrix)

    # Apply Truncated SVD
    svd = TruncatedSVD(n_components=n_components)
    reduced_matrix = svd.fit_transform(distance_matrix_scaled)

    # Explained variance
    explained_variance = svd.explained_variance_ratio_.sum()
    print(f"Explained variance with {n_components} components: {explained_variance}")