import train
import model
import wals
import auroc
import faster_auroc
import numpy as np
import datetime
import rec_results



TESTSET_RATIO=0.1
 
LOG_RATINGS = 0
LINEAR_RATINGS = 1


#DATA_FILE='lastfm/user_artist_data.txt'
DATA_FILE='lastfm_data/user_artists_log.dat'
#DATA_FILE='lastfm_data/user_artists_log_reduced.dat'
#DATA_FILE='lastfm_data/user_artists.dat'
#DATA_FILE='ml-latest-small/ratings.csv'
#DATA_FILE='ml-latest/ratings.csv'


CURRENT_TIME=datetime.datetime.now().strftime("%y%m%d_%H%M%S")
MODEL_DIR='model/'+CURRENT_TIME
DEFAULT_PARAMS = {
    'use_weight': True,
	'latent_factors': 5,
	'num_iters': 20,
	'reg': 0.07,
	'unobserved_weight': 0.01,
	'weight_type': LOG_RATINGS,
    'feature_weight_lin': 130.0,
    'feature_weight_exp': 0.08
}

OPTIMIZED_PARAMS = {
    'use_weight': True,
    'latent_factors': 20,
    'num_iters': 10,
	'reg': 0.07,
	'unobserved_weight': 0.001,
    'weight_type': LINEAR_RATINGS,
    'feature_weight_lin': 130.0,
    'feature_weight_exp': 0.08
}


MAX_NUM_OF_RECOMMEND_ITEMS=10


if __name__=='__main__':
	print('===============pipeline===============')


	CV=10
	cv_total=[]
	for cv in range(CV):
		print('CV[%d]' %cv)
		this_cv=[]
		print('data constructing...')
		already_rated_idx,user_map,item_map,train_sparse,test_sparse=model.ratings_train_test(DATA_FILE)
	
	
#		best_auroc=0
#		best_params={}
		for reg in [0.07]:
			for unobserved_weight in [0.01]:
				for latent_factor in [2,3,4,5,10,20]:
					for num_iters in [20]:
						for feature_weight_lin in [130]:
							DEFAULT_PARAMS['unobserved_weight']=unobserved_weight
							DEFAULT_PARAMS['reg']=reg
							DEFAULT_PARAMS['latent_factors']=latent_factor
							DEFAULT_PARAMS['num_iters']=num_iters
							DEFAULT_PARAMS['feature_weight_lin']=feature_weight_lin
							DEFAULT_PARAMS['weight_type']=1
							
							row_factor,col_factor=train.train(DEFAULT_PARAMS,train_sparse)
							mean_auroc=faster_auroc.calculate_auroc(row_factor,col_factor,already_rated_idx,test_sparse,len(item_map))
							DEFAULT_PARAMS['auroc']=mean_auroc
							
							threshold=0.0
							if mean_auroc > threshold : print(DEFAULT_PARAMS)
							this_cv.append(DEFAULT_PARAMS.copy())

#		DEFAULT_PARAMS=best_params
#		print('=======best=======')
#		print(DEFAULT_PARAMS)
#		print(best_auroc)
#

		import time

#		auc_list=[]
#		print(DEFAULT_PARAMS)
#		for _ in range(10):
#	
#			start_time=time.time()
##	print('training...')
#			row_factor,col_factor=train.train(DEFAULT_PARAMS,train_sparse)
#			elapsed_time = time.time() - start_time
##	print('training takes %f' % elapsed_time)	
#
#	start_time = time.time()
#	print('auroc calculating...')
#	_,auroc=auroc.calculate_total_auroc(row_factor,col_factor,already_rated_idx,test_sparse,len(item_map))
#	print(auroc)
#	elapsed_time = time.time() - start_time
#	print('auroc takes %f' % elapsed_time)
#
#			start_time = time.time()
#			auroc=faster_auroc.calculate_auroc(row_factor,col_factor,already_rated_idx,test_sparse,len(item_map))
#			print('AUROC[%f]' %auroc)
#			auc_list.append(auroc)
#			elapsed_time = time.time() - start_time
##	print('faster auroc takes %f' % elapsed_time)	
#		print('AVG[%f]' %np.mean(auc_list))
#		print('MAX[%f]' %max(auc_list))
#		print('MIN[%f]' %min(auc_list))
		cv_total.append(this_cv)
	
	auroc_matrix=[]
	for cv in cv_total:
		temp_list=[ params['auroc'] for params in cv]
		auroc_matrix.append(temp_list)

	auroc_matrix=np.array(auroc_matrix)
	print(auroc_matrix)
	mean_=np.mean(auroc_matrix,axis=0)
	max_=np.max(auroc_matrix,axis=0)
	min_=np.min(auroc_matrix,axis=0)
	print('AVG : %s' %mean_)
	print('MAX : %s' %max_)
	print('MIN : %s' %min_)
	

	for idx in range(len(mean_)):
		thres=0.0
		if mean_[idx] > thres:
			print(cv_total[0][idx])
			print('AVG[%f]' %mean_[idx])
			print('MAX[%f]' %max_[idx])
			print('MIN[%f]' %min_[idx])
		
			
	
	
	
	print('making recommendations...')
	rec_result=rec_results.make_rec_results(already_rated_idx,row_factor,col_factor,user_map,item_map,MAX_NUM_OF_RECOMMEND_ITEMS)
	
	assert False
	
	print('model saving...')
	model.save_model(MODEL_DIR, already_rated_idx, user_map, item_map, row_factor, col_factor)
	rec_results.save_rec_results(MODEL_DIR,rec_result)

	print('done')

