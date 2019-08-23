import numpy as np
from collections import defaultdict

import pickle 



ITEM_ID_NAME_FILE_NAME='lastfm_data/artists.dat'

def load_item_id_name_dict():
	itemDict={}
	with open(ITEM_ID_NAME_FILE_NAME,'rt') as f:
		f.readline()
		for lines in f.readlines():
			line=lines.split('\t')
			itemDict[int(line[0])]=line[1]

	return itemDict

		


def most_common_item_idx(already_rated_idx,n_items):
	item_count=np.zeros(n_items)
	for item_idxs in already_rated_idx:
		for item_idx in item_idxs:
			item_count[item_idx]+=1

	frequent_item_idx=np.argsort(item_count)
	frequent_item_idx=frequent_item_idx[::-1]
	
	
	return frequent_item_idx

def make_rec_results(already_rated_idx,row_factor,col_factor,user_map,item_map,k):
	
	item_id_name_dict=load_item_id_name_dict()
	rec_results=defaultdict(list)
	
	item_idx_name_dict=[ item_id_name_dict[item_id] for item_id in item_map]
		
	
	pred_results=row_factor.dot(col_factor.T)
	k_list = [len(rated_item_idx_list)+k for rated_item_idx_list in already_rated_idx]
	
	for user_idx in range(pred_results.shape[0]):
		rec_list_idx=np.argsort(pred_results[user_idx])[-k_list[user_idx]:]
		rec_list_idx=[item_idx for item_idx in rec_list_idx if item_idx not in already_rated_idx[user_idx]]
		rec_list_idx=rec_list_idx[-k:]
		rec_list_idx.reverse()


		user_id=user_map[user_idx]
		rec_results[user_id]=[item_idx_name_dict[item_idx] for item_idx in rec_list_idx]		

	most_common_item_idxs=most_common_item_idx(already_rated_idx,len(item_map))[:k]
	rec_results['fresh']=[item_idx_name_dict[item_idx] for item_idx in most_common_item_idxs]
	return rec_results


def save_rec_results(model_dir,rec_result):
	file_dir=model_dir+'/rec_results.pkl'
	with open (file_dir,'wb') as f:
		pickle.dump(rec_result,f)



