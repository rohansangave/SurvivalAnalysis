# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 00:00:09 2019

@author: pvsha
"""

import lifelines
import pandas as pd

from lifelines import CoxPHFitter
from lifelines.datasets import load_rossi
from lifelines.utils import concordance_index


df = pd.read_csv('./datasets/whas1638.csv',sep=',')
dt = df.values


rossi = load_rossi()



cph = CoxPHFitter()
cph.fit(df[['1', '2', '3', '4', '5', 'lenfol', 'fstat']], duration_col="lenfol", event_col="fstat")

cph.print_summary()
print(cph.score_)

