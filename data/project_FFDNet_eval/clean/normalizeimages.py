import numpy as np
from glob import glob

files = glob('*.npy')
M = 10.089038980848645
m = -1.429329123112601
for n in range(len(files)):
	im = np.load(files[n])
	print(np.max(im))
	logim = np.log(im+np.spacing(1))
	nlogim = ((logim-m)*255 / (M - m))
	nlogim = nlogim.astype('float32')
	filename = './lognormdata/'+files[n]
	np.save(filename,nlogim)
