""""Image compression using kmeam clustering
Part of Simplearn Machine Learning Course
Date: 20.10.2010
Done By Sofien Abidi"""

# Import standard libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#% matplotlib inline

# Import Dataset
from sklearn.datasets import load_sample_image

china = load_sample_image('china.jpg')
ax = plt.axes(xticks=[], yticks=[])
ax.imshow(china)
china.shape
data = china / 255.0
data = data.reshape(427 * 640, 3)


# print(data)
def plot_pixels(data, title, colors=None, N=10000):
    if colors is None:
        colors = data
    # choose a random subset
    rng = np.random.RandomState(0)
    i = rng.permutation(data.shape[0])[:N]
    colors = colors[i]
    R, G, B = data[i].T

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    ax[0].scatter(R, G, color=colors, marker='.')
    ax[0].set(xlabel='Red', ylabel='Green', xlim=(0, 1), ylim=(0, 1))

    ax[1].scatter(R, B, color=colors, marker='.')
    ax[1].set(xlabel='Red', ylabel='Blue', xlim=(0, 1), ylim=(0, 1))

    fig.suptitle(title, size=20)


plot_pixels(data, title='Input color space: 16 millions possible colors')

# Fix numpy error
import warnings;

warnings.simplefilter('ignore')

# Reducing these 16 millions colors to 16
from sklearn.cluster import MiniBatchKMeans

kmeans = MiniBatchKMeans(16)
kmeans.fit(data)
new_colors = kmeans.cluster_centers_[kmeans.predict(data)]

plot_pixels(data, colors=new_colors, title="reduced color space: 16 Colors")

# Recoloring original image
china_recolored = new_colors.reshape(china.shape)
ax = plt.axes(xticks=[], yticks=[])
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
fig.subplots_adjust(wspace=0.05)
ax[0].imshow(china)
ax[0].set_title('Original image', size=16)
ax[1].imshow(china_recolored)
ax[1].set_title('Compressed image', size=16)
plt.show()