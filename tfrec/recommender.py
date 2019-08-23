import os
import numpy as np
import time
import pickle
from collections import defaultdict

DEFAULT_REC_LIST=[]

#MODEL_DIR='model/190812_134740'
#MODEL_DIR='model/190812_133756'
MODEL_DIR='model/190812_133257'
#MODEL_DIR='model/190809_165643'
#MODEL_DIR='data_construct_results'
RATING_FILE='lastfm_data/user_artists_log.dat'

class recommender():


	
	
	def __init__(self, model_dir=MODEL_DIR):

#		self.load_model(model_dir)
		self.load_rec_results(model_dir)

	
	def load_rec_results(self,model_dir):
		with open (os.path.join(model_dir,'rec_results.pkl'),'rb') as f:
			self.rec_results=pickle.load(f)


	def load_model(self,model_dir):
		self.already_rated_idx=np.load(os.path.join(model_dir,'already_rated_idx.npy'))
		self.user_map=np.load(os.path.join(model_dir,'user_map.npy'))
		self.item_map=np.load(os.path.join(model_dir,'item_map.npy'))
		self.row_factor=np.load(os.path.join(model_dir,'row_factor.npy'))
		self.col_factor=np.load(os.path.join(model_dir,'col_factor.npy'))


	def get_recommendation(self,user_id,k):
		return self.rec_results[user_id][:k]
	def get_recommendation_(self,user_id,k):
		
	
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
	


	def predict(self,X,feature_names):


		try:	
			uid=int(X[0][0])
		except ValueError:
			return ['INVALID','USER','ID']

		rec_result=self.rec_results[uid]

		if not rec_result:
			print('THIS IS FRESH USER')
			return self.rec_results['fresh']
		else:
			return rec_result
			
def already_liked_items(rating_file):

	K=10
	already_liked_item=defaultdict(list)
		
	with open(rating_file,'r') as f:
		for lines in f.readlines():
			line=lines.split('\t')
			already_liked_item[int(line[0])].append((int(line[1]),float(line[2])))

	ITEM_ID_NAME_FILE='lastfm_data/artists.dat'
	def load_item_id_name_dict():
		itemDict={}
		with open(ITEM_ID_NAME_FILE,'rt') as f:
			f.readline()
			for lines in f.readlines():
				line=lines.split('\t')
				itemDict[int(line[0])]=line[1]

		return itemDict
	
	item_dict=load_item_id_name_dict()
	
	for user_id, ratings in already_liked_item.items():
		k=min(K,len(ratings))
		ratings_=sorted(ratings, key=lambda x:x[1],reverse=True)[:k]
#		already_liked_item[user_id]=ratings_
		already_liked_item[user_id]=[item_dict[item_id] for (item_id, _) in ratings_]

	
	return already_liked_item



if __name__=='__main__':
	
	rec_unit=recommender()


	already_liked_items=already_liked_items(RATING_FILE)
	
	import re
	for uid in range(2000,2002):
		
		X=np.array([[uid]])
		already_liked_item_list=already_liked_items[uid]
		hanCount=0
		recommended_items=rec_unit.predict(X,'feature_name')
		print('[USER ID : %d]\n%s\n%s' %(uid,already_liked_item_list,recommended_items))
		for item in already_liked_item_list:
			hanCount +=len(re.findall(u'[\u3130-\u318F\uAC00-\uD7A3]+', item))
			
		if hanCount>0:
			recommended_items=rec_unit.predict(X,'feature_name')
			print('[USER ID : %d]\n%s\n%s' %(uid,already_liked_item_list,recommended_items))
