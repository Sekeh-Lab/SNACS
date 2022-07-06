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
data[0] = np.asarray([1161.514694, 6573.510980, 14517.923324, 74125.9916, 271880.6285])/60./60.
data[1] = np.asarray([45.568682, 205.255498, 839.932627, 2439.358155, 9999.001388])/60./60.
data[2] = np.asarray([46.560894, 171.063943, 832.468836, 3961.957919, 10028.283979])/60/60.

event   = ['MST-based [Ganesh et al. 2020]', 'ACMI ($\\varphi=$1)', 'ACMI ($\\varphi=$weight)']
x_axis  = [16, 32, 64, 128, 256]

## Full plot
fig, ax = plt.subplots(figsize=(14, 9))
plt.plot(x_axis, data[1], '--^', linewidth=4, markersize=15, label=event[1])
plt.plot(x_axis, data[2], '--o', linewidth=4, markersize=15, label=event[2])
plt.plot(x_axis, data[0], '-*', linewidth=4, markersize=15, label=event[0])
plt.annotate('24x', (x_axis[0],data[0][0]), textcoords='offset points', xytext=(0,15), ha='center', fontsize=25)
plt.annotate('32x', (x_axis[1],data[0][1]), textcoords='offset points', xytext=(0,15), ha='center', fontsize=25)
plt.annotate('17x', (x_axis[2],data[0][2]), textcoords='offset points', xytext=(0,15), ha='center', fontsize=25)
plt.annotate('18x', (x_axis[3],data[0][3]), textcoords='offset points', xytext=(-5,15), ha='center', fontsize=25)
plt.annotate('27x', (x_axis[4],data[0][4]), textcoords='offset points', xytext=(-15,5), ha='center', fontsize=25)

plt.title('Estimator run-time comparison', fontsize=40)
plt.xlabel('Group Size', fontsize=35)
plt.ylabel('Run Time (hours)', fontsize=35)

plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.legend(fontsize=40)

plt.show()


## Full plot
#combined_data = {}
#combined_data['signal'] = data[1].flatten().tolist() + data[2].flatten().tolist() + data[0].flatten().tolist() 
#combined_data['event']  =  [names[1]]*int(len(combined_data['signal'])/3) + [names[2]]*int(len(combined_data['signal'])/3) + [names[0]]*int(len(combined_data['signal'])/3)
#combined_data['time']   = []
#time_axis               = [16, 32, 64, 128, 256]
#time_axis               = [np.repeat(i,10).tolist() for i in time_axis] + [np.repeat(i,10).tolist() for i in time_axis] +[np.repeat(i,10).tolist() for i in time_axis] 
#combined_data['time']   = [item for e1 in time_axis for item in e1]
#
#sns.lineplot(x='time', y='signal', data=combined_data, hue='event', markers=True, color=palette)
#
#
#plt.title('Estimator run-time comparison', fontsize=35)
#plt.xlabel('Group Size', fontsize=30)
#plt.ylabel('Run Time (hours)', fontsize=30)
#
#plt.xticks(fontsize=25)
#plt.yticks(fontsize=25)
#plt.legend(fontsize='30')
#
#plt.show()

## Only SNACS plot
#combined_data = {}
#combined_data['signal'] = data[1].flatten().tolist() + data[2].flatten().tolist()
#combined_data['event']  = [names[1]]*int(len(combined_data['signal'])/2) + [names[2]]*int(len(combined_data['signal'])/2)
#combined_data['time']   = []
#time_axis               = [16, 32, 64, 128, 256]
#time_axis               = [np.repeat(i,10).tolist() for i in time_axis] +[np.repeat(i,10).tolist() for i in time_axis] 
#combined_data['time']   = [item for e1 in time_axis for item in e1]
#
#sns.lineplot(x='time', y='signal', data=combined_data, hue='event', markers=True)
#
#
#plt.title('Estimator run-time comparison', fontsize=35)
#plt.xlabel('Group Size', fontsize=30)
#plt.ylabel('Run Time (hours)', fontsize=30)
#
#plt.xticks(fontsize=25)
#plt.yticks(fontsize=25)
#plt.legend(fontsize='30')
#
#plt.show()
