import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess
from kneed import KneeLocator

import constants
import utils



# Fit a linear regression model and display evaluation metrics and diagnostic plots if required.
def fit_linear_reg_customized(X, y, label_reg='', show_eval_metrics=False):
    """
    Fits a linear regression model and optionally displays evaluation metrics and diagnostic plots.

    Parameters:
    X : array-like or DataFrame
        The input features for the regression model.
    y : array-like or Series
        The target variable for the regression model.
    label_reg : str, optional (default='')
        The label for the regression model, used in plot titles and printed output.
    show_eval_metrics : bool, optional (default=False)
        If True, prints evaluation metrics and shows diagnostic plots.

    Returns:
    sm_model : statsmodels.regression.linear_model.RegressionResultsWrapper
        The fitted linear regression model.
    
    Evaluation Metrics and Diagnostic Plots:
    If show_eval_metrics is True, the function prints the model summary and generates a 2x2 grid of diagnostic plots:
        1. Residuals vs Fitted Values Plot
        2. QQ Plot of studentized residuals
        3. Scale-Location Plot (Spread-Location)
        4. Influence Plot
    """

    # Fitting the linear regression model with statsmodels
    sm_model = sm.OLS(y, X).fit()

    #---------------------------------------------------------
    # Evaluation metrics and plots
    if show_eval_metrics:

        print('--------------------------------------------------------------------')
        print(f'Regression assessment metrics for {label_reg} regression:')
        print(sm_model.summary())
        print('\n\n\n')

        # Creating a 2x2 grid for subplots
        eval_fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        eval_fig.tight_layout(pad=5.0)  # Adjust spacing between subplots

        # Set a general title for the evaluation plots figure
        eval_fig.suptitle(f'Evaluation Plots for {label_reg} regression', fontsize=16)

        # 1. Residuals Plot
        residuals = sm_model.resid
        smoothed_residuals = lowess(residuals, sm_model.fittedvalues, frac=0.1)
        axs[0, 0].scatter(sm_model.fittedvalues, residuals, alpha=0.5)
        axs[0, 0].plot(smoothed_residuals[:, 0], smoothed_residuals[:, 1], color='red', label='Smoothed Line')
        axs[0, 0].axhline(0, color='black', linestyle='--', linewidth=2)
        axs[0, 0].set_title('Residuals vs Fitted Values')
        axs[0, 0].set_xlabel('Fitted Values')
        axs[0, 0].set_ylabel('Residuals')

        # 2. QQ Plot
        standardized_residuals = sm_model.get_influence().resid_studentized_internal
        sm.qqplot(standardized_residuals, line='45', ax=axs[0, 1])
        axs[0, 1].set_title('QQ Plot of studentized residuals')

        # 3. Scale-Location Plot
        smoothed_residuals = lowess(np.sqrt(np.abs(standardized_residuals)), sm_model.fittedvalues, frac=0.1)
        axs[1, 0].scatter(sm_model.fittedvalues, np.sqrt(np.abs(standardized_residuals)),  alpha=0.5)
        axs[1, 0].plot(smoothed_residuals[:, 0], smoothed_residuals[:, 1], color='red', label='Smoothed Line')
        axs[1, 0].set_title('Scale-Location Plot (Spread-Location)')
        axs[1, 0].set_xlabel('Fitted Values')
        axs[1, 0].set_ylabel('Sqrt |Studentized Residuals|')

        # 4. Leverage Plot
        sm.graphics.influence_plot(sm_model, criterion="cooks", ax=axs[1, 1])
        axs[1, 1].set_title('Influence Plot')
    
    return sm_model



# Function fo flag outliers based on the prediction intervals of a linear regression
def flag_outliers_linear_models(df, x_variable, y_variable, label_reg='', alpha=0.05, show_eval_metrics=False):
    """
    Identifies outliers in a linear regression model based on a specified prediction interval.

    This function fits a linear regression model using the specified X and y variables,
    computes the prediction interval bounds, and flags observations as outliers if their
    residuals fall outside the prediction interval bounds.

    Parameters:
    df : DataFrame
        The input DataFrame containing the data.
    x_variable : str
        The name of the independent variable (X) for the regression model.
    y_variable : str
        The name of the dependent variable (y) for the regression model.
    label_reg : str, optional (default='')
        The label for the regression model, used in plot titles and printed output.
    alpha : float, optional (default=0.05)
        The significance level for the prediction interval. Default is 0.05 (95% confidence interval).
    show_eval_metrics : bool, optional (default=False)
        If True, prints evaluation metrics and shows diagnostic plots.

    Returns:
    DataFrame
        A copy of the original DataFrame with an additional column 'outlier_lin_reg'
        indicating whether each observation is an outlier ('yes') or not ('no').
    linear_model : statsmodels.regression.linear_model.RegressionResultsWrapper
        The fitted linear regression model.
    """
         
    # Make a copy to avoid modofying the original DataFrame
    df_outliers = df.copy()

    # Extract X and y values for the fit_linear_reg_customized function
    X = df_outliers[[x_variable]].values
    y = df_outliers[y_variable].values

    # Adding a constant to X (the statsmodels library needs it for the intercept)
    X_with_const = sm.add_constant(X)

    # Getting the fitted model
    linear_model = fit_linear_reg_customized(X=X_with_const, 
                                            y=y,
                                            label_reg=label_reg,
                                            show_eval_metrics=show_eval_metrics)

    #-----------------------------------------------------------------------------------------------
    # Flagging the outliers as observations with residuals that fall outside the prediction interval
    # I need to compute the bounds for each observation
    pred_interval_bounds_flag = linear_model.get_prediction(X_with_const).summary_frame(alpha=alpha) # Prediction interval bounds -(1-alpha)% confidence- 
    y_lower_flag = pred_interval_bounds_flag['obs_ci_lower'] 
    y_upper_flag = pred_interval_bounds_flag['obs_ci_upper']

    # I need to reset the indices of the original dataframe and the outlier bounds
    # because the outlier bounds indices do not coincide with those of the original data
    df_outliers = df_outliers.reset_index(drop=True)
    y_lower_flag = y_lower_flag.reset_index(drop=True)
    y_upper_flag = y_upper_flag.reset_index(drop=True)

    # Flagging the outliers
    df_outliers['outlier_lin_reg'] = 'no'
    outlier_mask = (df_outliers[y_variable] < y_lower_flag) | ( df_outliers[y_variable] > y_upper_flag)
    df_outliers.loc[outlier_mask, 'outlier_lin_reg'] = 'yes'

    return df_outliers, linear_model



def gmm_1d_boundaries(data, n_centers=2, bins_hist=500, random_seed=constants.RANDOM_SEED,
                      x_label='Measured variable',  filename_to_save=None):
    """
    Fit a 1D Gaussian Mixture Model (GMM) to the given data and visualize the components, along with their 95% coverage boundaries.

    Parameters
    ----------
    data : ndarray
        1D array of data points to fit the Gaussian Mixture Model.
    n_centers : int, optional
        Number of Gaussian components to fit in the model. Default is 2.
    bins_hist : int, optional
        Number of bins to use for the data histogram. Default is 500.
    random_seed : int, optional
        Seed for random number generation to ensure reproducibility. Default is `constants.RANDOM_SEED`.
    x_label : str, optional
        Label for the x-axis in the plot. Default is 'Measured variable'.
    filename_to_save : str or None, optional
        If specified, saves the generated plot to the provided filename. Default is None.

    Returns
    -------
    dict_results : dict
        A dictionary containing the following keys:
            - 'boundaries': List of tuples where each tuple represents the 95% coverage boundaries for a Gaussian component.
            - 'bic': Bayesian Information Criterion (BIC) for the fitted GMM.
    """

    # Settingup the figure size
    plt.figure(figsize=(8, 4))

    # Fit a Gaussian Mixture Model with 2 components
    gmm = GaussianMixture(n_components=n_centers, random_state=random_seed)
    gmm.fit(data)

    # Generate dynamic colors for each component
    colors = utils.generate_colors(gmm.n_components)

    # Plot the data histogram
    plt.hist(data, bins=bins_hist, density=True, alpha=0.4, color='black')

    # Create a range of values for the GMM PDF
    x = np.linspace(data.min(), data.max(), 1000).reshape(-1, 1)

    # Variables to return the results
    dict_results = {}
    list_boundaries = []

    # Sorting the components by their means in ascending order (to facilitate visualization)
    sorted_indices = np.argsort(gmm.means_.flatten())

    # Plot individual Gaussians corresponding to each component
    for i, color in zip(sorted_indices, colors):
        mean = gmm.means_[i, 0]
        cov = gmm.covariances_[i, 0, 0]
        weight = gmm.weights_[i]
        std_dev = np.sqrt(cov)

        # Individual Gaussian PDF
        pdf_individual = weight * (1 / np.sqrt(2 * np.pi * cov)) * np.exp(-(x - mean)**2 / (2 * cov))
        plt.plot(x, pdf_individual, label=f'Gaussian {sorted_indices.tolist().index(i)+1}', color=color)

        # Compute 95% boundaries
        boundary_lower = mean - 1.96 * std_dev
        boundary_upper = mean + 1.96 * std_dev

        # Mark 95% boundaries
        plt.axvline(boundary_lower, color=color, linestyle=':', alpha=0.6, label='95% boundaries')
        plt.axvline(boundary_upper, color=color, linestyle=':', alpha=0.6)

        # Add text for boundaries with background
        plt.text(boundary_lower, 0.01, f"{boundary_lower:.2f}", color='white', ha='right',
                fontsize=11, bbox=dict(facecolor=color, alpha=0.7, edgecolor='none'))
        plt.text(boundary_upper, 0.01, f"{boundary_upper:.2f}", color='white', ha='left',
                fontsize=11, bbox=dict(facecolor=color, alpha=0.7, edgecolor='none'))
        
        # Adding the tuple of boundaries to the list
        list_boundaries.append((boundary_lower, boundary_upper))

    # Plot the total GMM fit (combined Gaussian)
    logprob = gmm.score_samples(x)
    pdf = np.exp(logprob)
    plt.plot(x, pdf, label='GMM fit', color='black', linestyle='--', linewidth=2)

    # Customize the plot
    plt.title(f"1D Gaussian Mixture Model with {n_centers} components")
    plt.xlabel(x_label)
    plt.ylabel('Density')
    plt.legend()

    # Save the plot if requested
    if filename_to_save:
        plt.savefig(filename_to_save)

    plt.show()

    dict_results['boundaries'] = list_boundaries
    dict_results['bic'] = gmm.bic(data)

    return dict_results



# Function to compute the elbow method with the Gaussian Mixture Model
def gmm_elbow_method(data, max_components=10, random_seed=constants.RANDOM_SEED, filename_to_save=None):
    """
    Perform model selection for a 1D Gaussian Mixture Model (GMM) using the Elbow Method with AIC and BIC scores.

    Parameters
    ----------
    data : ndarray
        1D array of data points to fit the Gaussian Mixture Models.
    max_components : int, optional
        Maximum number of Gaussian components to consider. Default is 10.
    random_seed : int, optional
        Seed for random number generation to ensure reproducibility. Default is `constants.RANDOM_SEED`.
    filename_to_save : str or None, optional
        If specified, saves the AIC/BIC plot to the provided filename. Default is None.

    Returns
    -------
    None
    """
    
    aic_scores = []
    bic_scores = []
    models = []

    # Fit GMMs for different numbers of components and calculate AIC/BIC
    components = range(1, max_components + 1)
    for n in components:
        gmm = GaussianMixture(n_components=n, random_state=random_seed)
        gmm.fit(data)
        aic_scores.append(gmm.aic(data))
        bic_scores.append(gmm.bic(data))
        models.append(gmm)

    # Plot AIC and BIC scores
    plt.figure(figsize=(3.5, 4))
    plt.plot(components, aic_scores, label='AIC', marker='o')
    plt.plot(components, bic_scores, label='BIC', marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Score')
    plt.title('GMM model selection using AIC and BIC')
    plt.legend()
    plt.grid(True)

    # Save the plot if requested
    if filename_to_save:
        plt.savefig(filename_to_save)

    plt.show()

    # Find the optimal number of components based on minimum AIC/BIC
    optimal_aic = np.argmin(aic_scores) + 1
    optimal_bic = np.argmin(bic_scores) + 1

    print(f"Optimal number of components based on AIC: {optimal_aic}")
    print(f"Optimal number of components based on BIC: {optimal_bic}")

    # Find the elbow point
    elbow_aic = KneeLocator(x=list(components), y=aic_scores, curve='convex', direction='decreasing')
    elbow_bic = KneeLocator(x=list(components), y=bic_scores, curve='convex', direction='decreasing')

    print(f"Elbow point based on AIC: {elbow_aic.elbow}")
    print(f"Elbow point based on BIC: {elbow_bic.elbow}")



def sequence_df_to_numpy(sequences_df, feature_columns, sequence_identifiers=['uid', 'tid']):
    """
    This function groups the input DataFrame by the specified sequence identifiers, extracts the 
    feature columns, and converts the grouped sequences into a 3D NumPy array with shape 
    (n_sequences, n_timesteps, n_features). Shorter sequences are padded with np.nan to match the 
    length of the longest sequence.

    Args:
        sequences_df (pandas.DataFrame): The input DataFrame containing sequences.
        feature_columns (list): A list of column names in the DataFrame to be used as features.
        sequence_identifiers (list, optional): A list of column names to group the sequences by.
                                               Default is ['uid', 'tid'].

    Returns:
        numpy.ndarray: A 3D NumPy array with shape (n_sequences, n_timesteps, n_features) containing
                       the sequences data, where n_sequences is the number of unique sequences, 
                       n_timesteps is the length of the longest sequence, and n_features is the 
                       number of feature columns. Shorter sequences are padded with NaNs.
    """
    sequences_grouped = sequences_df.groupby(sequence_identifiers)[feature_columns]
    sequences_list = [group.to_numpy() for _, group in sequences_grouped]

    # Compute the array dimensions
    n_sequences = len(sequences_list)
    n_timesteps = max(map(len,sequences_list)) # The array has the length of the longest sequence
    n_features = len(feature_columns)

    # Create an empty array with shape (n_sequences, n_timesteps, n_features)
    sequences_array = np.full((n_sequences, n_timesteps, n_features), np.nan, dtype=np.float64)

    # Fill the array with the sequences data
    for i, sequence in enumerate(sequences_list):
        sequences_array[i, :len(sequence), :] = sequence  # Assign only existing time steps. Shorter trajectories are padded with np.nan from creation

    return sequences_array