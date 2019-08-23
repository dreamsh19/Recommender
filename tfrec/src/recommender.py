import os
import numpy as np
import time
import sys

DEFAULT_REC_LIST=[]

class recommender():


	def __init__(self, model_dir):

		self.load_model(model_dir)


	def load_model(self,model_dir):
		self.already_rated_idx=np.load(os.path.join(model_dir,'already_rated_idx.npy'))
		self.user_map=np.load(os.path.join(model_dir,'user_map.npy'))
		self.item_map=np.load(os.path.join(model_dir,'item_map.npy'))
		self.row_factor=np.load(os.path.join(model_dir,'row_factor.npy'))
		self.col_factor=np.load(os.path.join(model_dir,'col_factor.npy'))

	def get_recommendation(self,user_id,k):
		
	
		user_idx=np.searchsorted(self.user_map,user_id)	
		

		rec_list=[]
		if self.user_map[user_idx]==user_id :
			already_rated_idx=self.already_rated_idx[user_idx]
			
			# I think github source code is wrong ( # is orginial code )
			# assert self.row_factor.shape[0]-len(already_rated_idx) >=k

			assert self.col_factor.shape[0]-len(already_rated_idx) >=k

			user_factor=self.row_factor[user_idx]
			pred_ratings=self.col_factor.dot(user_factor)
			k_=k+len(already_rated_idx)
			
			rec_list_idx=np.argsort(pred_ratings)[-k_:]
			rec_list_idx=[ item_idx for item_idx in rec_list_idx if item_idx not in already_rated_idx]

			rec_list_idx=rec_list_idx[-k:]
			rec_list_idx.reverse()

			rec_list=[ self.item_map[item_idx] for item_idx in rec_list_idx]

		return rec_list
			


def main(argv):
	
	start_time=time.time()

	print('argv')
	print(argv)
	print('model loading...')
	rec_unit=recommender('model/190802_161203')

	print('load_time',time.time()-start_time)

	k=10
	
	uid=int(argv)
	
	for USER_ID in range(uid,uid+1):
		print(USER_ID)
		start_time=time.time()
		rec_list=rec_unit.get_recommendation(USER_ID,k)
		print('rec_time',time.time()-start_time)
		print(rec_list)


if __name__=='__main__':
		main(sys.argv[1])
