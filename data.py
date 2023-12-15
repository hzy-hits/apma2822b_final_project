import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import glob

directory ="data"
filenames=[]

for i in range(0, 99):
    filepath = f"data/result_{i}.txt"
    data=np.loadtxt(filepath)
    # Plot and save the image
    plt.imshow(data, interpolation='nearest')
    plt.colorbar()  # Add the color bar
    data=data[200:1400,200:1400]
    # Save the figure
    filename = f'temp_plot_{i}.png'
    plt.savefig(filename)
    filenames.append(filename)
    plt.close()
    i=i+1
with Image.open(filenames[0]) as img:
    img.save('heatmap_animation.gif', save_all=True, append_images=[Image.open(f) for f in filenames[1:]], duration=300, loop=0)

# 清理临时文件
for filename in filenames:
    os.remove(filename)
