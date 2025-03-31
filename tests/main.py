import src
from pathlib import Path

directory = Path(__file__).resolve().parent

wind_data = src.get_data(directory)

    