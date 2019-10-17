import os
import logging.config

from pathlib import Path

import dotenv

# Load the .env entries as environment variables
dotenv.load_dotenv()

PROJECT_DIR = Path(__file__).resolve().parents[1]

# load logger config
logging.config.fileConfig(
    PROJECT_DIR.joinpath('logging.ini'),
    disable_existing_loggers=False)

# Set up directory paths; allow overriding from env
DATA_DIR = Path(os.getenv('MLH_DATA_DIR', PROJECT_DIR.joinpath('data')))
OUT_DIR = Path(os.getenv('MLH_OUT_DIR', PROJECT_DIR.joinpath('out')))
MODELS_DIR = Path(PROJECT_DIR.joinpath('models'))
