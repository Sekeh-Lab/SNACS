import seaborn           as sns
import numpy             as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import matplotlib

plt.rcParams["font.family"] = "Times New Roman"

# Setup Colo Palette
#palette = clr.ListedColormap(['#003f5c', '#2f4b7c', '#665191', '#a05195', '#d45087', '#f95d6a', '#ff7c43', '#ffa600'])
plt.set_cmap('viridis')
cmap = matplotlib.cm.get_cmap('viridis')
color_range = np.linspace(0, 1, 7)

palette = sns.color_palette('tab10', 3)

data = {}
## ResNet50
## MINT Compression
#data[0] = np.asarray([0.0, 73.46, 31.90, 44.14, 34.37, 3.12, 28.51, 27.02, 17.57, 26.58, 35.27, 48.51, 27.85, 36.13, 83.32, 24.56, 29.02, 33.76, 43.84, 51.75, 14.18, 38.23, 32.98, 32.98, 84.76, 35.74, 81.32, 77.44, 10.74, 12.06, 52.97, 46.67, 90.84, 88.52, 66.94, 25.80, 21.09, 48.63, 22.31, 4.12, 32.44, 28.24, 30.15, 3.46, 5.22, 91.06, 84.22, 67.38, 61.64, 52.02, 89.11, 77.95, 68.45, 0.0]) 
## SNACS compression
#data[1] = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 11.20, 0.02, 0.02, 0.02, 0.02, 16.04, 6.05, 16.04, 21.04, 26.04, 6.05, 16.04, 26.04, 16.04, 26.04, 21.04, 0.02, 6.05, 26.04, 56.05, 36.05, 36.05, 51.04, 41.04, 41.04, 41.04, 21.04, 36.05, 51.04, 21.04, 16.04, 61.05, 36.05, 16.04, 46.04, 41.04, 21.04, 46.04, 66.04, 88.28, 61.05, 41.04, 46.04, 95.60, 66.04, 96.04, 96.04, 0.0])
#x_axis  = np.arange(0, 54) 

# VGG16 
# MINT Compression
data[0] = np.asarray([0.0, 0.0, 0.0, 0.0, 67.08, 51.80, 58.39, 61.18, 89.91, 93.72, 92.04, 86.25, 91.01, 93.28, 0.0]) 
# SNACS compression
data[1] = np.asarray([0.0, 0.0, 0.0, 21.04, 51.04, 71.04, 86.05, 87.25, 91.04, 96.04, 91.04, 91.04, 66.04, 93.35, 0.15])
x_axis  = np.arange(0, 15) 

## ResNet56 
## MINT Compression
#data[0] = np.asarray([0.0, 19.53, 4.68, 6.64, 13.67, 4.29, 0, 17.18, 0, 13.67, 3.12, 15.23, 6.25, 19.14, 8.20, 0, 17.18, 1.56, 26.56, 0, 22.26, 34.86, 38.76, 63.57, 24.60, 38.18, 31.54, 67.18, 16.69, 26.75, 29, 47.16, 31.83, 38.57, 26.23, 41.6, 25.58, 0, 66.47, 79.76, 71.50, 78.61, 79.95, 88.96, 72.29, 73.55, 81.71, 82.61, 71.77, 81.49, 73.65, 63.35, 25.19, 0, 20.04, 0]) 
## SNACS compression
#data[1] = np.asarray([0.0, 0.39, 0.39, 0.39, 51.56, 11.32, 0.39, 0.39, 16.40, 16.40, 11.32, 6.25, 21.48, 6.25, 26.56, 41.40, 51.56, 0.39, 26.56, 11.32, 36.13, 51.07, 36.13, 66.11, 41.11, 26.07, 76.07, 31.15, 46.09, 16.11, 26.07, 26.07, 21.09, 6.15, 46.09, 11.13, 26.07, 41.11, 76.07, 46.04, 91.04, 56.05, 81.05, 11.05, 86.05, 36.06, 86.05, 31.05, 91.04, 91.04, 96.04, 96.04, 96.04, 96.04, 96.04, 0.15])
#x_axis  = np.arange(0, 56) 

event   = ['[Ganesh et al. 2020]', 'SNACS (ours)']

## Full plot
fig, ax = plt.subplots(figsize=(15, 6.7))
plt.bar(x_axis- 0.45/2., data[0], 0.45, alpha=0.5, color='g', label=event[0])
plt.bar(x_axis+ 0.45/2., data[1], 0.45, label=event[1])

#plt.xlim(-1,55)
plt.xlim(-1,14)

#plt.title('Comparison of compression ($\%$) per layer', fontsize=30)
plt.xlabel('Layer', fontsize=25)
plt.ylabel('Compression ($\%$)', fontsize=25)

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=25)
#plt.tight_layout()
#plt.margins(0,0)
plt.show()
