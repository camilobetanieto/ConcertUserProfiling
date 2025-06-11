import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASETS_PATH = Path(os.getenv("DATASETS_PATH"))
INPUT_PATH = DATASETS_PATH / "Input"
PROCESSED_PATH = DATASETS_PATH / "Processed"
LOCATIONS_SPLIT_FILTERED_PATH = PROCESSED_PATH / "wifi_traces_filtered_split"
PREPROCESSED_TRAJECTORIES_PATH = PROCESSED_PATH / "preprocessed_trajectories"
TRAJECTORIES_EVENTS_PATH = PROCESSED_PATH / "trajectories_events"


REPORTS_PATH = PROJECT_ROOT / "reports"
FIGURES_PATH = REPORTS_PATH / "figures"