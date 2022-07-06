import seaborn           as sns
import numpy             as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import matplotlib


# Setup Colo Palette
#palette = clr.ListedColormap(['#003f5c', '#2f4b7c', '#665191', '#a05195', '#d45087', '#f95d6a', '#ff7c43', '#ffa600'])
plt.set_cmap('viridis')
cmap = matplotlib.cm.get_cmap('viridis')
color_range = np.linspace(0, 1, 7)

palette = sns.color_palette('tab10', 3)

# Setup Data
names   = ['MINT', 'SNACS ($\phi=1$)', 'SNACS ($\phi=weight$)']
colors  = ['blue', 'coral', 'green']
data    = {}
data[0] = np.asarray([[1148.546686, 1157.107768, 1160.366887, 1170.977844, 1180.558351, 1155.430290, 1153.496039, 1170.719519, 1155.267678, 1162.675875], 
                     [6389.09029627, 6597.62294412, 6550.94573069, 6583.92176461, 6607.08636022, 6503.52694917, 6686.30688334, 6504.31365228, 6617.91644931, 6694.37877417], 
                     [21449.17872691, 18575.19383168, 20249.9985311, 16703.93233514, 11365.86757517, 11944.67103672, 11883.84336829, 11233.84420323, 10141.61121345, 11631.09242272], 
                     [72080.408245, 70098.414959, 73076.173986, 68111.585872, 66977.640881, 76109.653194, 82990.909063, 83563.146633, 83563.146633, 83563.146633], 
                     [294985.087544, 248776.169638, 248776.169638, 248776.169638, 248776.169638, 248776.169638, 248776.169638, 248776.169638, 248776.169638, 248776.169638]])/60/60.

data[1] = np.array([[45.4893229, 45.61784244, 45.61027813, 45.14472914, 45.73908877, 45.11686778, 45.1526165, 46.13817191, 45.40842128, 46.2694838],
                   [202.56019711, 204.68826127, 206.70101762, 204.84643006, 206.04273891, 205.64232278, 205.72905397, 204.28340435, 206.91281176, 205.14874268],
                   [842.60869694, 838.66726136, 837.35505795, 836.05384898, 838.33151793, 848.91337967, 836.19927335, 842.89704132, 839.09679937, 839.20339441],
                   [2428.71203661, 2441.4944768, 2436.94582891, 2439.16980004, 2438.25741434, 2442.25756383, 2435.38271928, 2445.19708419, 2444.46695137, 2441.69767499],
                   [9963.08346081, 9989.48337936, 9995.386482, 10005.69969201, 9970.63631487, 10028.33027434, 10001.16612458, 10014.58813214, 10015.49071503, 10006.14930081]])/60./60.

data[2] = np.array([[46.21907496, 46.21766734, 46.48890638, 46.4897306, 46.34571815, 47.33559823, 46.54712939, 47.32209039, 46.46556664, 46.17745495],
                   [171.51184106, 168.97651529, 171.92197657, 168.97500587, 171.8542676, 171.79183793, 171.24521422, 172.58111954, 170.71115088, 171.07050347],
                   [829.79477859, 833.53639817, 830.45964599, 827.99124312, 830.13986373, 833.26914072, 837.00194645, 831.89106441, 835.85144496, 834.75283313],
                   [4743.98654652, 4254.79880786, 3709.1016686, 3994.27263808, 3899.62052393, 3532.58982205, 3820.65253043, 3948.12376404, 3676.0450058, 4040.38788319],
                   [12588.75857854, 11218.821661, 11827.13562369, 9865.55971098, 8796.576226, 9028.50677705 , 8747.54452014, 9846.74991536, 8704.75794697, 9658.42882633]])/60/60.

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

# Only SNACS plot
combined_data = {}
combined_data['signal'] = data[1].flatten().tolist() + data[2].flatten().tolist()
combined_data['event']  = [names[1]]*int(len(combined_data['signal'])/2) + [names[2]]*int(len(combined_data['signal'])/2)
combined_data['time']   = []
time_axis               = [16, 32, 64, 128, 256]
time_axis               = [np.repeat(i,10).tolist() for i in time_axis] +[np.repeat(i,10).tolist() for i in time_axis] 
combined_data['time']   = [item for e1 in time_axis for item in e1]

sns.lineplot(x='time', y='signal', data=combined_data, hue='event', markers=True)


plt.title('Estimator run-time comparison', fontsize=35)
plt.xlabel('Group Size', fontsize=30)
plt.ylabel('Run Time (hours)', fontsize=30)

plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.legend(fontsize='30')

plt.show()
