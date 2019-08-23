import recommender

if __name__=='__main__':

	dictPath='prediction_dict.pkl'
	n=10
	rc=recommender.recommender(dictPath,n)
	rc.train()
	print('train.py complete')
