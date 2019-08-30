import os
import numpy as np
import time
import pickle
from collections import defaultdict

DEFAULT_REC_LIST=[]


MODEL_DIR='data_construct_results'
RATING_FILE='lastfm_data/user_artists_log.dat'

class recommender():

	
	def __init__(self, model_dir=MODEL_DIR):

		self.load_rec_results(model_dir)

	
	def load_rec_results(self,model_dir):
		with open (os.path.join(model_dir,'rec_results.pkl'),'rb') as f:
			self.rec_results=pickle.load(f)


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
		already_liked_item[user_id]=[item_dict[item_id] for (item_id, _) in ratings_]

	
	return already_liked_item



if __name__=='__main__':
	
	rec_unit=recommender()

	already_liked_items=already_liked_items(RATING_FILE)
	
	for uid in range(2000,2002):
		
		X=np.array([[uid]])
		already_liked_item_list=already_liked_items[uid]
		recommended_items=rec_unit.predict(X,'feature_name')
		print('[USER ID : %d]\n%s\n%s' %(uid,already_liked_item_list,recommended_items))
