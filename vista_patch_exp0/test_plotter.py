


import numpy as np

import matplotlib.pyplot as plt

from skimage import io




lai_see = np.load('./dataset/france2/processed_lai_npy1/Q_LAI_2022_measure_10.npy')

print("np.unique(lai_see)", np.unique(lai_see))

print("lai_see.max(), lai_see.min()", lai_see.max(), lai_see.min())


plt.figure(figsize=(12, 12))
plt.subplot(232)
plt.title(' Ground truth')
plt.imshow(lai_see)
plt.axis('off')
plt.savefig('./view_check/seeit.png', bbox_inches='tight')
plt.close()