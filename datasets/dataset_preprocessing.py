import h5py
from collections import defaultdict
import pandas as pd

def load_datasets(dataset_file):
    datasets = defaultdict(dict)

    with h5py.File(dataset_file, 'r') as fp:
        for ds in fp:
            for array in fp[ds]:
                datasets[ds][array] = fp[ds][array][:]

    return datasets

def format_dataset_to_df(dataset, duration_col, event_col, trt_idx = None):
    xdf = pd.DataFrame(dataset['x'])
    if trt_idx is not None:
        xdf = xdf.rename(columns={trt_idx : 'treat'})

    dt = pd.DataFrame(dataset['t'], columns=[duration_col])
    censor = pd.DataFrame(dataset['e'], columns=[event_col])
    cdf = pd.concat([xdf, dt, censor], axis=1)
    return cdf


dataset = load_datasets("whas_train_test.h5")
df1 = format_dataset_to_df(dataset['train'], "lenfol", "fstat")
df2 = format_dataset_to_df(dataset['test'], "lenfol", "fstat")
df = df1.append(df2, ignore_index=True)
print(df.head())
print(df.shape)
df.to_csv("whas1638.csv", index=False)
