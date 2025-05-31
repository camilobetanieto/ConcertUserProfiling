from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
import numpy as np
from scipy.stats import entropy
import dbcv

class KMedoidsExperiment:
    """
    A class to perform K-medoids clustering experiments with multiple distance matrices and number of clusters (k).

    This class runs K-medoids clustering on precomputed distance matrices, evaluates clustering quality 
    using silhouette scores, and identifies the best clustering parameters.

    Attributes:
        multi_distances_embeddings (dict): A dictionary where keys are lambda values and values are dictionaries containing 
                                           'distances' (precomputed distance matrices) and 'embeddings' (the corresponding data embeddings).
        k_values (list of int): A list of k values to use for K-medoids.
        labels_dict (dict): A dictionary storing cluster labels for each (lambda, k) pair.
        silhouette_scores_dict (dict): A dictionary storing silhouette scores for each clustering result.
        dbcv_scores_dict (dict): A dictionary storing DBCV scores for each clustering result.
        best_parameters_silhouette (tuple or None): The (lambda, k) pair with the highest silhouette score.
        best_score_silhouette (float or None): The highest silhouette score achieved.
    """

    def __init__(self, multi_distances_embeddings, k_values):
        """
        Initializes the KMedoidsExperiment instance.

        Args:
            multi_distances_embeddings (dict): A dictionary where keys are lambda values and values are 
                                               dictionaries containing 'distances' (precomputed distance matrices) 
                                               and 'embeddings' (the corresponding data embeddings).
            k_values (list of int): A list of k values to use for KMedoids.
        """
        self.multi_distances_embeddings = multi_distances_embeddings
        self.k_values = k_values

        self.labels_dict = {}
        self.silhouette_scores_dict = {}
        self.dbcv_scores_dict = {}
        self.rescaled_silhouette_scores_dict = {}
        self.normalized_entropies_dict = {}
        self.ranked_balanced_clustering_scores_dict = {}

        self.best_parameters_silhouette = None
        self.best_score_silhouette = None
        self.best_parameters_bcs = None
        self.best_score_bcs = None




    def run_multi_kmedoids(self):
        """
        Runs K-medoids clustering for each combination of distance matrix and k.

        The resulting cluster labels are stored in `labels_dict` with 
        (lambda, k) as keys.

        Updates:
            - `labels_dict`: Stores cluster labels for each (lambda, k) combination.
        """
        for lambda_val, distances_embeddings in self.multi_distances_embeddings.items():
            embeddings_copy = distances_embeddings['embeddings'].copy()
            for k in self.k_values:
                kmedoids = KMedoids(n_clusters=k, metric='euclidean')
                kmedoids.fit(embeddings_copy)
                output_labels = kmedoids.labels_

                self.labels_dict[(lambda_val, k)] = output_labels
            print(f'Performed clustering for combined distance with lambda = {lambda_val}')
        print(f'Clustering is done. K-medoids labels_dict was updated.')




    def compute_kmedoids_silhouette_scores(self):
        """
        Computes the silhouette scores for each clustering result in `labels_dict`.

        Stores the results in `silhouette_scores_dict` with (lambda, k) as keys.

        Updates:
            - `silhouette_scores_dict`: Stores silhouette scores for each clustering configuration.
        """
        if not self.labels_dict:
            raise ValueError("The labels_dict is empty. Scores cannot be computed.")

        for (lambda_val, k), output_labels in self.labels_dict.items():
            # Get unique labels
            unique_labels = np.unique(output_labels)

            # Compute silhouette score only if there are at least 2 clusters
            if len(unique_labels) >= 2:
                embeddings = self.multi_distances_embeddings[lambda_val]['embeddings']
                score = silhouette_score(embeddings, output_labels, metric='euclidean')
                self.silhouette_scores_dict[(lambda_val, k)] = score
            else:
                print(f"Skipping (lambda={lambda_val}, k={k}) - only {len(unique_labels)} cluster(s) found.")
                self.silhouette_scores_dict[(lambda_val, k)] = np.nan

        print(f'Silhouette scores computation is done. K-medoids silhouette_scores_dict was updated')




    def compute_kmedoids_dbcv_scores(self):
        """
        Computes the DBCV scores for each clustering result in `labels_dict`.

        Stores the results in `dbcv_scores_dict` with (lambda, k) as keys.

        Updates:
            - `dbcv_scores_dict`: Stores DBCV scores for each clustering configuration.
        """
        if not self.labels_dict:
            raise ValueError("The labels_dict is empty. Scores cannot be computed.")
    
        for (lambda_val, k), output_labels in self.labels_dict.items():
            embeddings = self.multi_distances_embeddings[lambda_val]['embeddings']
            dbcv_score = dbcv.dbcv(X=embeddings, y=output_labels)
            self.dbcv_scores_dict[(lambda_val, k)] = dbcv_score

        print(f'DBCV scores computation is done. K-medoids dbcv_scores_dict was updated')




    def compute_normalized_entropy(self):
        """
        Computes the normalized entropy of the clustering labels.
        Normalization occurs to the range [0, 1].

        Updates:
        - `normalized_entropies_dict`: A dictionary mapping
          (lambda_val, k) to their respective normalized entropy scores.
        """
        if not self.labels_dict:
            raise ValueError("The labels_dict is empty. Entropies cannot be computed.")

        for (lambda_val, k), labels in self.labels_dict.items():
            _, counts = np.unique(labels, return_counts=True)
            label_probabilities = counts / len(labels)
            entropy_value = entropy(label_probabilities, base=2)
            max_entropy = np.log2(len(counts))  # Maximum entropy (when the distribution is uniform)
            normalized_entropy = entropy_value / max_entropy  # Normalize to [0, 1]

            self.normalized_entropies_dict[(lambda_val, k)] = normalized_entropy
        print(f'Normalized entropy computation is done. K-medoids normalized_entropies_dict was updated')




    def rescale_silhouette_scores(self):
        """
        Rescale the silhouette scores to the range [0, 1].

        The silhouette score ranges from -1 to 1, 
        where higher values indicate better clustering quality. This method rescales 
        the scores using the transformation: (silhouette + 1) / 2.

        Updates:
            `rescaled_silhouette_scores_dict`: A dictionary mapping
          (lambda_val, k) to their respective 
          rescaled silhouette values.
        """
        if not self.silhouette_scores_dict:
            raise ValueError("The silhouette_scores_dict is empty. Rescaling cannot be computed.")
        
        for (lambda_val, k), silhouette_score in self.silhouette_scores_dict.items():
            rescaled_silhouette = (silhouette_score + 1) / 2
            
            self.rescaled_silhouette_scores_dict[(lambda_val, k)] = rescaled_silhouette
        print(f'Rescaling of silhouette is done. K-medoids rescaled_silhouette_scores_dict was updated')




    def compute_balanced_clustering_score(self, weight=0.5):
        """
        Compute a balanced clustering score using a weighted combination of rescaled silhouette and normalized entropy.

        The formula used is:
            balanced_score = (weight * rescaled_silhouette_scores_dict) + ((1 - weight) * entropy_score)

        Args:
            weight (float, optional): A value between 0 and 1 that controls the balance between 
                                    rescaled DBCV and entropy. Default is 0.5.

        Updates:
            - `ranked_balanced_clustering_scores_dict`: A sorted dictionary
            containing (lambda_val, k) and their corresponding balanced score, ranked in descending order.
        """
        if not self.rescaled_silhouette_scores_dict:
            raise ValueError("The self.rescaled_silhouette_scores_dict is empty. Balanced clustering scores cannot be computed.")
        
        if not self.normalized_entropies_dict:
            raise ValueError("The normalized_entropies_dict is empty. Balanced clustering scores cannot be computed.")
        
            
        # Compute combined metric
        combined_scores = {}
        for (lambda_val, k), rescaled_silhouette_score in self.rescaled_silhouette_scores_dict.items():
            entropy_score = self.normalized_entropies_dict[(lambda_val, k)] # Has the same keys as rescaled_silhouette_score_dict

            combined_score = (weight * rescaled_silhouette_score) + ((1 - weight) * entropy_score) 
            combined_scores[(lambda_val, k)] = combined_score
        
        # Rank the combinations based on the combined score
        self.ranked_balanced_clustering_scores_dict = dict(sorted(combined_scores.items(), key=lambda x: x[1], reverse=True))
        print(f'Balanced clustering scores computation and ranking is done. K-medoids ranked_balanced_clustering_scores_dict was updated')




    def update_best_params_and_score(self):
        """
        Finds the (lambda, k) combinations that achieve 
        the highest silhouette and BCS (balanced clustering score) values,
        and updates `best_parameters_silhouette`, and `best_parameters_bcs` accordingly.

        Updates:
            - `best_parameters_silhouette`: The (lambda, k) combination with the highest silhouette score.
            - `best_score_silhouette`: The highest silhouette score achieved.
            - `best_parameters_bcs`: The (lambda, k) combination with the highest balanced clustering score.
            - `best_score_bcs`: The highest balanced clustering score achieved.
        """
        self.best_parameters_silhouette = max(self.silhouette_scores_dict, key=self.silhouette_scores_dict.get)
        self.best_score_silhouette = self.silhouette_scores_dict[self.best_parameters_silhouette]
        print(f'K-medoids updated best_parameters are {self.best_parameters_silhouette} with silhouette score = {self.best_score_silhouette}')

        self.best_parameters_bcs = max(self.ranked_balanced_clustering_scores_dict, key=self.ranked_balanced_clustering_scores_dict.get)
        self.best_score_bcs = self.ranked_balanced_clustering_scores_dict[self.best_parameters_bcs]
        print(f'K-medoids updated best_parameters_bcs are {self.best_parameters_bcs} with BCS score = {self.best_score_bcs}')
