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


df = pd.read_csv('whas500.csv',sep=',')
dt = df.values


rossi = load_rossi()



cph = CoxPHFitter()
cph.fit(df, duration_col="lenfol", event_col="fstat")

cph.print_summary()
print(cph.score_)

