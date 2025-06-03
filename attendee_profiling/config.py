import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
DATASETS_PATH = Path(os.getenv("DATASETS_PATH"))
PROCESSED_PATH = DATASETS_PATH / "Processed"
LOCATIONS_SPLIT_FILTERED_PATH = PROCESSED_PATH / "wifi_traces_filtered_split"