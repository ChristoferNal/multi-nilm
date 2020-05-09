from multiprocessing.pool import Pool

import numpy as np
import pandas as pd


def parallelize_dataframe(df, func, num_partitions, num_cores):
    df_split = np.array_split(df, num_partitions)
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df
