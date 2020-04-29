import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec

# read David stool data

df_ts = {}

files = ['25_timeseries.txt', '28_timeseries.txt']
keys = ['David_stool_A', 'David_stool_B']

for i, f in enumerate(files):
    x = pd.read_csv(f, na_values='NAN', delimiter='\t', header=None)
    
    x = x.T
    
    x.insert(0, 'time', range(len(x)))
    
    x.columns = ['time'] + ['species_%d' % j for j in range(1, len(x.columns))]
    
    df_ts[keys[i]] = x

# plot data

fig = plt.figure(figsize=(15,10))
gs = gridspec.GridSpec(2,1,hspace=0.4)

for i, key in enumerate(keys):
	ax = fig.add_subplot(gs[i])
	ax.set_title(key)
	ax.plot(df_ts[key].time, df_ts[key].values[:,1::5]) # plot 1 species in 5
	ax.set_yscale('log')
	ax.set_ylabel('Abundance')

plt.show()