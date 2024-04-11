import pathlib

import pandas as pd


df = pd.read_csv(
    pathlib.Path(__file__).parent.parent.parent.parent/'data'/'data.csv',
    index_col=0
)
