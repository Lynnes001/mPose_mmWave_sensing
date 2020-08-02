#%%

from utility import featureExtraction
import numpy as np
import glob, os

# Step 1. Extract heatmaps from ADC data

envFolder = "..//data//"

# mk output dir
if not os.path.isdir(envFolder+'data_features//'):
    os.mkdir(envFolder+'data_features//')

# loop files, extract heatmaps.
data_folders = glob.glob(envFolder+"data_raw//*")
for i, folder in enumerate(data_folders):
    if i > -1:
        filename = folder.split('/')[5]
        folder = folder+'//'
        print(filename, ' ', str(i+1)+'//'+str(len(data_folders)))
        [heatmaps, timestampList, rangeReso] = featureExtraction(folder, envFolder+'data_features//'+filename, i)

# output heatmaps are saved to ..//data//data_features//


#%% Visualize heatmaps

import matplotlib.pyplot as plt
import glob, os
import numpy as np

heatmapsFiles = glob.glob("..//data//data_features//*")

heatmapsSet = np.load(heatmapsFiles[0])
heatmaps = heatmapsSet["heatmaps"]
ts = heatmapsSet["timestampList"]

plt.imshow(heatmaps[0])
plt.show()


# %%
