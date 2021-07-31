#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import sys


def ensemble(filenames):
    if len(filenames) != 3:
        print("There were not 3 filenames given, do not forget to separate filenames with commas.")
        return
    filename1 = filenames[0]
    filename2 = filenames[1]
    filename3 = filenames[2]
    df1 = pd.read_csv(filename1)
    df2 = pd.read_csv(filename2)
    df3 = pd.read_csv(filename3)
    df_ens = (df1["prediction"] + df2["prediction"] + df3["prediction"])/3
    df_ens[df_ens < 0.5] = 0
    df_ens[df_ens >= 0.5] = 1
    df5 = df1
    df5["prediction"] = df_ens.astype(np.int64)
    df5.to_csv("ensemble_preds.csv", index=False)


file_list= sys.argv[1].split(',')
ensemble(file_list)

