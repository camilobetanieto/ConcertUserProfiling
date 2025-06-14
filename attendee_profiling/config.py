import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASETS_PATH = Path(os.getenv("DATASETS_PATH"))
INPUT_PATH = DATASETS_PATH / "Input"
PROCESSED_PATH = DATASETS_PATH / "Processed"
CLIPPED_POLYGONS_PATH = PROCESSED_PATH / "Zonas SONAR clipped"
LOCATIONS_SPLIT_FILTERED_PATH = PROCESSED_PATH / "wifi_traces_filtered_split"
PREPROCESSED_TRAJECTORIES_PATH = PROCESSED_PATH / "preprocessed_trajectories"
TRAJECTORIES_EVENTS_PATH = PROCESSED_PATH / "trajectories_events"
MODEL_INPUTS_PATH = PROCESSED_PATH / "model_inputs_preprocessed"
TABLES_DESCRIPTION_PATH = PROCESSED_PATH / "tables_for_description"
DISTANCES_EMBEDDINGS_PATH = PROCESSED_PATH / "distances_embeddings"
CLUSTER_DESCRIPTION_PATH = PROCESSED_PATH / "8-clusters-description"
CLUSTERING_RESULTS_PATH = PROCESSED_PATH / "clustering_results"
GRAPH_FILES_PATH = PROCESSED_PATH / "graph_files"


REPORTS_PATH = PROJECT_ROOT / "reports"
FIGURES_PATH = REPORTS_PATH / "figures"
DESCRIPTIVE_ANALYSIS_PLOTS_PATH = FIGURES_PATH / "6-descriptive-analysis-plots"