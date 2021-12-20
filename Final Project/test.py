import pandas as pd
import numpy as np

dates = pd.date_range('20130101', periods=6)
df = pd.DataFrame(np.arange(24).reshape((6,4)), columns=['A','B','C','D'])

print(df)

df.iloc[2,2] = 1111
df.loc[0,'B'] = 2222

print(df)
