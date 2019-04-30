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

dataset = load_datasets("gaussian_survival_data.h5")
# dataset = load_datasets("linear_survival_data.h5")
# dataset = load_datasets("whas_train_test.h5")
# dataset = load_datasets("support_train_test.h5")
# dataset = load_datasets("metabric_IHC4_clinical_train_test.h5")
# dataset = load_datasets("gbsg_cancer_train_test.h5")
# print(dataset)
df1 = format_dataset_to_df(dataset['train'], "lenfol", "fstat")
print(len(df1.index))
df2 = format_dataset_to_df(dataset['valid'], "lenfol", "fstat")
print(len(df2.index))
df3 = format_dataset_to_df(dataset['test'], "lenfol", "fstat")
print(len(df3.index))
df = df1.append(df2, ignore_index=True)
df = df.append(df3, ignore_index=True)
print(df.head())
print(len(df.index))
# df.to_csv("support8873.csv", index=False)
# df.to_csv("metabric.csv", index=False)
# df.to_csv("gbsg.csv", index=False)
# df.to_csv("linear_survival_data.csv", index=False)
df.to_csv("gaussian_survival_data.csv", index=False)
