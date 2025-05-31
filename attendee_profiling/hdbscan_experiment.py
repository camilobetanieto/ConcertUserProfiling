#from sklearn.cluster import HDBSCAN
import hdbscan
from sklearn.metrics import silhouette_score
import numpy as np
import dbcv
from scipy.stats import entropy

class HDBSCANExperiment:
    """
    A class to perform HDBSCAN clustering experiments with multiple distance matrices and cluster sizes.

    This class runs HDBSCAN clustering on precomputed distance matrices, evaluates clustering quality 
    using silhouette and DBCV scores, and identifies the best clustering parameters.

    Attributes:
        multi_distances_embeddings (dict): A dictionary where keys are the lambda values and the values are dictionaries containing 
                                           'distances' (precomputed distance matrices) and 'embeddings' (the corresponding data embeddings).
        min_cluster_sizes (list of int): A list of minimum cluster sizes to use for HDBSCAN.
        min_samples_proportions (list of float): A list of proportions to determine the minimum samples for HDBSCAN.
        labels_dict (dict): A dictionary storing cluster labels for each (lambda, min_cluster_size, min_samples_proportion) pair.
        silhouette_scores_dict (dict): A dictionary storing silhouette scores for each clustering result.
        dbcv_scores_dict (dict): A dictionary storing DBCV scores for each clustering result.
        best_parameters_silhouette (tuple or None): The (lambda, min_cluster_size, min_samples_proportion) combination 
                                                    with the highest silhouette score.
        best_score_silhouette (float or None): The highest silhouette score achieved.
        best_parameters_dbcv (tuple or None): The (lambda, min_cluster_size, min_samples_proportion) combination 
                                              with the highest DBCV score.
        best_score_dbcv (float or None): The highest DBCV score achieved.
    """

    def __init__(self, multi_distances_embeddings, min_cluster_sizes, min_samples_proportions):
        """
        Initializes the HDBSCANExperiment instance.

        Args:
            multi_distances_embeddings (dict): A dictionary where keys are lambda values and values are 
                                               dictionaries containing 'distances' (precomputed distance matrices) 
                                               and 'embeddings' (the corresponding data embeddings).
            min_cluster_sizes (list of int): A list of minimum cluster sizes to use for HDBSCAN.
            min_samples_proportions (list of float): A list of proportions to determine the minimum samples for HDBSCAN.
        """
        self.multi_distances_embeddings = multi_distances_embeddings
        self.min_cluster_sizes = min_cluster_sizes
        self.min_samples_proportions = min_samples_proportions

        self.labels_dict = {}
        self.hdbscan_instances_dict = {}
        self.silhouette_scores_dict = {}
        self.dbcv_scores_dict = {}
        self.rescaled_dbcv_scores_dict = {}
        self.normalized_entropies_dict = {}
        self.ranked_balanced_clustering_scores_dict = {}

        self.best_parameters_silhouette = None
        self.best_score_silhouette = None
        self.best_parameters_dbcv = None
        self.best_score_dbcv = None
        self.best_parameters_bcs = None
        self.best_score_bcs = None

        



    def run_multi_hdbscan(self, hdbscan_params={}):
        """
        Runs HDBSCAN clustering for each combination of distance matrix, min_cluster_size, and min_samples_proportion.

        The resulting cluster labels are stored in `labels_dict` with 
        (lambda, min_cluster_size, min_samples_proportion) as keys.

        Parameters:
        hdbscan_params (dict, optional): Additional parameters to be passed to the HDBSCAN algorithm. 
                                         Defaults to an empty dictionary.

        Updates:
            - `labels_dict`: Stores cluster labels for each (lambda, min_cluster_size, min_samples_proportion) combination.
        """
        for lambda_val, distances_embeddings in self.multi_distances_embeddings.items():
            embeddings_copy = distances_embeddings['embeddings'].copy()
            for min_cluster_size in self.min_cluster_sizes:
                for min_samples_proportion in self.min_samples_proportions:
                    min_samples = round(min_cluster_size*min_samples_proportion)
                    hdbscan_instance = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                                                       metric='euclidean', **hdbscan_params)
                    output_labels = hdbscan_instance.fit_predict(embeddings_copy)

                    # Store the instance and the labels
                    self.hdbscan_instances_dict[(lambda_val, min_cluster_size, min_samples_proportion)] = hdbscan_instance
                    self.labels_dict[(lambda_val, min_cluster_size, min_samples_proportion)] = output_labels
            print(f'Performed clustering for combined distance with lambda = {lambda_val}')
        print(f'Clustering is done. HDBSCAN labels_dict was updated.')





    def compute_hdbscan_silhouette_scores(self):
        """
        Computes the silhouette scores for each clustering result in `labels_dict`.

        Stores the results in `silhouette_scores_dict` with (lambda, min_cluster_size, min_samples_proportion) as keys.

        Updates:
            - `silhouette_scores_dict`: Stores silhouette scores for each clustering configuration.
        """
        if not self.labels_dict:
            raise ValueError("The labels_dict is empty. Scores cannot be computed.")
        
        for (lambda_val, min_cluster_size, min_samples_proportion), output_labels in self.labels_dict.items():
            # Get unique labels
            unique_labels = np.unique(output_labels)

            # Compute silhouette score only if there are at least 2 clusters
            if len(unique_labels) >= 2:
                embeddings = self.multi_distances_embeddings[lambda_val]['embeddings']
                score = silhouette_score(embeddings, output_labels, metric='euclidean')
                self.silhouette_scores_dict[(lambda_val, min_cluster_size, min_samples_proportion)] = score
            else:
                print(f"Skipping (lambda={lambda_val}, min_cluster_size={min_cluster_size}), min_samples_proportion={min_samples_proportion} - only {len(unique_labels)} cluster(s) found.")
                self.silhouette_scores_dict[(lambda_val, min_cluster_size, min_samples_proportion)] = np.nan

        print(f'Silhouette scores computation is done. HDBSCAN silhouette_scores_dict was updated')




    def compute_hdbscan_dbcv_scores(self):
        """
        Computes the DBCV scores for each clustering result in `labels_dict`.

        Stores the results in `dbcv_scores_dict` with (lambda, min_cluster_size, min_samples_proportion) as keys.

        Updates:
            - `dbcv_scores_dict`: Stores DBCV scores for each clustering configuration.
        """
        if not self.labels_dict:
            raise ValueError("The labels_dict is empty. Scores cannot be computed.")
    
        for (lambda_val, min_cluster_size, min_samples_proportion), output_labels in self.labels_dict.items():
            embeddings = self.multi_distances_embeddings[lambda_val]['embeddings']
            dbcv_score = dbcv.dbcv(X=embeddings, y=output_labels)
            self.dbcv_scores_dict[(lambda_val, min_cluster_size, min_samples_proportion)] = dbcv_score

        print(f'DBCV scores computation is done. HDBSCAN dbcv_scores_dict was updated')




    def compute_normalized_entropy(self):
        """
        Computes the normalized entropy of the clustering labels.
        Normalization occurs to the range [0, 1].

        Excludes noise points (label = -1) before computing entropy.

        Updates:
        - `normalized_entropies_dict`: A dictionary mapping
          (lambda_val, min_cluster_size, min_samples_proportion) to their respective 
          normalized entropy scores.
        """
        if not self.labels_dict:
            raise ValueError("The labels_dict is empty. Entropies cannot be computed.")

        for (lambda_val, min_cluster_size, min_samples_proportion), labels in self.labels_dict.items():
            labels = labels[labels != -1] # Exclude noise
            _, counts = np.unique(labels, return_counts=True)
            label_probabilities = counts / len(labels)
            entropy_value = entropy(label_probabilities, base=2)
            max_entropy = np.log2(len(counts))  # Maximum entropy (when the distribution is uniform)
            normalized_entropy = entropy_value / max_entropy  # Normalize to [0, 1]

            self.normalized_entropies_dict[(lambda_val, min_cluster_size, min_samples_proportion)] = normalized_entropy
        print(f'Normalized entropy computation is done. HDBSCAN normalized_entropies_dict was updated') 




    def rescale_dbcv_scores(self):
        """
        Rescale the DBCV scores to the range [0, 1].

        The DBCV (Density-Based Clustering Validation) score ranges from -1 to 1, 
        where higher values indicate better clustering quality. This method rescales 
        the scores using the transformation: (dbcv + 1) / 2.

        Updates:
            `rescaled_dbcv_scores_dict`: A dictionary mapping
          (lambda_val, min_cluster_size, min_samples_proportion) to their respective 
          rescaled DBCV values.
        """
        if not self.dbcv_scores_dict:
            raise ValueError("The dbcv_scores_dict is empty. Rescaling cannot be computed.")
        
        for (lambda_val, min_cluster_size, min_samples_proportion), dbcv_score in self.dbcv_scores_dict.items():
            rescaled_dbcv = (dbcv_score + 1) / 2
            
            self.rescaled_dbcv_scores_dict[(lambda_val, min_cluster_size, min_samples_proportion)] = rescaled_dbcv
        print(f'Rescaling of DBCV is done. HDBSCAN rescaled_dbcv_scores_dict was updated')




    def compute_balanced_clustering_score(self, weight=0.5):
        """
        Compute a balanced clustering score using a weighted combination of rescaled DBCV and normalized entropy.

        The formula used is:
            balanced_score = (weight * rescaled_dbcv_score) + ((1 - weight) * entropy_score)

        Args:
            weight (float, optional): A value between 0 and 1 that controls the balance between 
                                    rescaled DBCV and entropy. Default is 0.5.

        Updates:
            - `ranked_balanced_clustering_scores_dict`: A sorted dictionary
            containing (lambda_val, min_cluster_size, min_samples_proportion) 
            and their corresponding balanced score, ranked in descending order.
        """
        if not self.rescaled_dbcv_scores_dict:
            raise ValueError("The self.rescaled_dbcv_scores_dict is empty. Balanced clustering scores cannot be computed.")
        
        if not self.normalized_entropies_dict:
            raise ValueError("The normalized_entropies_dict is empty. Balanced clustering scores cannot be computed.")
        
            
        # Compute combined metric
        combined_scores = {}
        for (lambda_val, min_cluster_size, min_samples_proportion), rescaled_dbcv_score in self.rescaled_dbcv_scores_dict.items():
            entropy_score = self.normalized_entropies_dict[(lambda_val, min_cluster_size, min_samples_proportion)] # Has the same keys as rescaled_dbcv_scores_dict

            combined_score = (weight * rescaled_dbcv_score) + ((1 - weight) * entropy_score) 
            combined_scores[(lambda_val, min_cluster_size, min_samples_proportion)] = combined_score
        
        # Rank the combinations based on the combined score
        self.ranked_balanced_clustering_scores_dict = dict(sorted(combined_scores.items(), key=lambda x: x[1], reverse=True))
        print(f'Balanced clustering scores computation and ranking is done. HDBSCAN ranked_balanced_clustering_scores_dict was updated')




    def update_best_params_and_score(self):
        """
        Finds the (lambda, min_cluster_size, min_samples_proportion) combinations that achieve 
        the highest silhouette, DBCV and BCS (balanced clustering score) values,
        and updates `best_parameters_silhouette`, `best_parameters_dbcv` and `best_parameters_bcs` accordingly.

        Updates:
            - `best_parameters_silhouette`: The (lambda, min_cluster_size, min_samples_proportion) combination with the highest silhouette score.
            - `best_score_silhouette`: The highest silhouette score achieved.
            - `best_parameters_dbcv`: The (lambda, min_cluster_size, min_samples_proportion) combination with the highest DBCV score.
            - `best_score_dbcv`: The highest DBCV score achieved.
            - `best_parameters_bcs`: The (lambda, min_cluster_size, min_samples_proportion) combination with the highest balanced clustering score.
            - `best_score_bcs`: The highest balanced clustering score achieved.
        """
        self.best_parameters_silhouette = max(self.silhouette_scores_dict, key=self.silhouette_scores_dict.get)
        self.best_score_silhouette = self.silhouette_scores_dict[self.best_parameters_silhouette]
        print(f'HDBSCAN updated best_parameters_silhouette are {self.best_parameters_silhouette} with silhouette score = {self.best_score_silhouette}')

        self.best_parameters_dbcv = max(self.dbcv_scores_dict, key=self.dbcv_scores_dict.get)
        self.best_score_dbcv = self.dbcv_scores_dict[self.best_parameters_dbcv]
        print(f'HDBSCAN updated best_parameters_dbcv are {self.best_parameters_dbcv} with DBCV score = {self.best_score_dbcv}')

        self.best_parameters_bcs = max(self.ranked_balanced_clustering_scores_dict, key=self.ranked_balanced_clustering_scores_dict.get)
        self.best_score_bcs = self.ranked_balanced_clustering_scores_dict[self.best_parameters_bcs]
        print(f'HDBSCAN updated best_parameters_bcs are {self.best_parameters_bcs} with BCS score = {self.best_score_bcs}')
