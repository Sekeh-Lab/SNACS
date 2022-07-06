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
# Samples = 500, 1000, 5000, 10000, 15000, 20000, 25000, Dims = 1 1 2
data[0] = np.asarray([0.013949, 0.011235, 0.012011, 0.009772, 0.009533, 0.008856, 0.008773])
data[1] = np.asarray([0.010517, 0.007542, 0.005638, 0.003906, 0.003648, 0.003243, 0.003036])

# Samples = 5000 , Dims = 3, 10, 20, 30, 50
data[2] = np.asarray([0.082392, 0.196301, 0.257731, 0.301595, 0.329420])
data[3] = np.asarray([0.033571, 0.047066, 0.062571, 0.076653, 0.080025])

event   = ['ACMI ($\\varphi=$1)', 'ACMI ($\\varphi=\exp(-\\frac{||act||_2^2}{2})$)']
x_axis1 = [500, 1000, 5000, 10000, 15000, 20000, 25000]
x_axis2 = [3,   10,  20,   30,   50]

## Plot 1
fig, ax = plt.subplots(figsize=(14, 9))
plt.plot(x_axis1, data[0], '--^', linewidth=4, markersize=15, label=event[0])
plt.plot(x_axis1, data[1], '--+', linewidth=4, markersize=15, label=event[1])

plt.title('MSE vs. Samples', fontsize=40)
plt.xlabel('Number of Samples', fontsize=35)
plt.ylabel('MSE', fontsize=35)

plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.legend(fontsize=40)

plt.show()

## Plot 2
fig, ax = plt.subplots(figsize=(14, 9))
plt.plot(x_axis2, data[2], '--^', linewidth=4, markersize=15, label=event[0])
plt.plot(x_axis2, data[3], '--+', linewidth=4, markersize=15, label=event[1])

plt.title('MSE vs. Dimensionality', fontsize=40)
plt.xlabel('Dimensionality of Variables', fontsize=35)
plt.ylabel('MSE', fontsize=35)

plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.legend(fontsize=40)

plt.show()
