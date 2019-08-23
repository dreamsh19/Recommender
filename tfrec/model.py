import numpy as np
from scipy.sparse import coo_matrix
import pandas as pd
import os
from collections import defaultdict
import DataGenerator
TESTSET_RATIO=0.1


#SEP=' '
SEP='\t'
#SEP=','

NAMES=['USER_ID','ITEM_ID','RATING']
#NAMES=['USER_ID','ITEM_ID','RATING','TIMESTAMP']
#HEADER=None
HEADER=0

DTYPE={'USER_ID':np.int64,'ITEM_ID':np.int64,'RATING':np.float64}
#DTYPE={'USER_ID':np.int64,'ITEM_ID':np.int64,'RATING':np.float64,'TIMESTAMP':np.int64}


def ratings_train_test(data_file):	


	if not os.path.exists(data_file):
		DataGenerator.data_generate()
 
	ratings_df=pd.read_csv(data_file,sep=SEP,names=NAMES,header=HEADER,dtype=DTYPE)

	ratings=ratings_df.values
	_user_col=ratings_df['USER_ID'].values
	_item_col=ratings_df['ITEM_ID'].values

	unique_users=np.unique(_user_col)
	unique_items=np.unique(_item_col)

	n_users=unique_users.shape[0]
	n_items=unique_items.shape[0]
	
	max_user=unique_users[-1]
	max_item=unique_items[-1]

	z = np.zeros(max_user+1, dtype=int)
	z[unique_users] = np.arange(n_users)
	u_r=z[_user_col]
	

	z = np.zeros(max_item+1, dtype=int)
	z[unique_items]= np.arange(n_items)
	i_r=z[_item_col]	

	_rating_col=ratings_df['RATING'].values
	ratings=np.zeros((_rating_col.shape[0],3), dtype=object)
	ratings[:,0]=u_r
	ratings[:,1]=i_r

#	import math
#	_rating_col=[math.log10(r+1) for r in _rating_col]

	ratings[:,2]=_rating_col	
	

	already_rated=defaultdict(list)
	for u,i,r in ratings:
		already_rated[u].append(i)
	
	already_rated_idx=[]
	for u,i_list in already_rated.items():
		already_rated_idx.append(sorted(i_list))

	def create_sparse_train_test():	
#	def create_sparse_train_test(ratings,n_users,n_items):

		testset_size=int(len(ratings)*TESTSET_RATIO)
		testset_idx_list=np.random.choice(len(ratings),testset_size,replace=False)
	
		ratings_test=ratings[testset_idx_list]
		ratings_train=np.delete(ratings,testset_idx_list,axis=0)
	

		u,i,r=zip(*ratings_train)
		train_sparse=coo_matrix((r,(u,i)),shape=(n_users,n_items))

		u,i,r=zip(*ratings_test)
	
		test_sparse=coo_matrix((r,(u,i)),shape=(n_users,n_items))
		
		return train_sparse,test_sparse
		
#	train_sparse,test_sparse = create_sparse_train_test(ratings,n_users,n_items)
	train_sparse,test_sparse = create_sparse_train_test()

	return already_rated_idx, unique_users, unique_items ,train_sparse,test_sparse


def save_model(model_dir,already_rated_idx,user_map,item_map,row_factor,col_factor):

	os.makedirs(model_dir)
	np.save(os.path.join(model_dir, 'already_rated_idx'), already_rated_idx)
	np.save(os.path.join(model_dir, 'user_map'), user_map)
	np.save(os.path.join(model_dir, 'item_map'), item_map)
	np.save(os.path.join(model_dir, 'row_factor'), row_factor)
	np.save(os.path.join(model_dir, 'col_factor'), col_factor)
	
if __name__=='__main__':
	
	ratings=[ (u,i,r) for u in range(10) for i in range(5) for r in range(5)]
	ratings=np.array(ratings)
	n_users=10
	n_items=5
	
	ratings_train_test(DATA_FILE)
