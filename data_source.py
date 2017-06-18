import pandas as pd
import numpy as np

def get_7lc_winning_numbers():
    data = pd.read_excel("datasets/7lc.xls")
    winning_numbers_cnt = 8
    return data.values[:, 2:2+winning_numbers_cnt]


