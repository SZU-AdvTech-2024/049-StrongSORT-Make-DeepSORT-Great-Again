import os
import numpy as np

path='detections_MOT16' #一个文件夹下多个npy文件
txtpath='detections_MOT16'
namelist=[x for x in os.listdir(path)]
for i in range( len(namelist) ):
	datapath=os.path.join(path,namelist[i]) #specific address
	print(namelist[i])
	#data = np.load(datapath).reshape([-1, 2])  # (39, 2)
	data = np.load(datapath) # (39, 2)
	# data = input_data.reshape(1, -1)

	np.savetxt('%s/%s.txt'%(txtpath,namelist[i][:-4]),data)
print ('over')