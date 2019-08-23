import os
import pickle

import numpy as np

def load_file(model_dir,*args):
	
	objs=()
	for arg in args:
		file_name=os.path.join(model_dir,arg)
		if arg.split('.')[-1] == 'npy':
			obj=np.load(file_name)


		else:
			with open (file_name,'rb') as f:
				obj=pickle.load(f)

		objs=objs+(obj,)


	return objs




