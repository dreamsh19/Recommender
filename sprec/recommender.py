import surprise
from surprise import Dataset
from surprise import Reader
from collections import defaultdict
import numpy as np
import DataGenerator
import pickle
import os

DATA_DIRECTORY_PATH='./lastfm_data'
DATA_FILE_NAME='user_artists_log.dat'

class recommender:


	
	def __init__(self,dictPath='prediction_dict.pkl',n=10):
		
		self.data = None
		self.trainset = None
		self.testset = None
		self.algo = None
		self.predictions = None
		self.predictionDict = None
	
		self.dictPath=dictPath
		self.n=n

	def dataSetConstruct(self):
		
		np.random.seed(0)
		file_path = DATA_DIRECTORY_PATH[2:]+'/'+DATA_FILE_NAME
		
		if not os.path.exists(file_path):
			DataGenerator.data_generate()
		reader = Reader(line_format='user item rating', sep='\t')
		self.data = Dataset.load_from_file(file_path, reader=reader)
		print('Contructing trainset and testset')
		self.trainset = self.data.build_full_trainset()
		self.testset = self.trainset.build_anti_testset()

	def train(self):

		self.dataSetConstruct()
		print("model training...")
		self.algo = surprise.SVDpp(n_factors=1, n_epochs=300, lr_all=0.001, reg_all=0.01)
		surprise.model_selection.cross_validate(self.algo, self.data, measures=['RMSE', 'MAE'], cv=3, verbose=True)
		
		print("model training complete")
		print("Making predictions...")
		self.predictions=self.algo.test(self.testset)
		print("Predictions made")


		
		self.predictionDict=defaultdict(list)
		for uid, iid, true_r, est, _ in self.predictions:
			self.predictionDict[uid].append((iid,est))
		
		print("Sorting results...")
		for id, ratings in self.predictionDict.items():
			self.predictionDict[id]=sorted(ratings,key=lambda x:x[1], reverse=True)[0:self.n]
		print("Sorting complete")
	

		file=open(self.dictPath,'wb')
		pickle.dump(self.predictionDict,file)
		file.close()
		print('Dict saved')
		

			

	def getResult(self,uid):
		if self.predictionDict is None:
			file=open(self.dictPath,'rb')
			self.predictionDict=pickle.load(file)
			file.close()
		arr=[]
		
		ratings=self.predictionDict.get(str(uid))
		if ratings is None:
			print('INVALID USER ID')
			return arr
	
		for tuple in self.predictionDict.get(str(uid)):
			arr.append(tuple[0])
		
		return arr
	
	def predict(self,X,feature_names):
		
		return self.getResult(int(X[0][0]))
	
if __name__=='__main__':
	uid=np.array([[2,]],dtype=np.float64)
	temp=recommender()
	print(temp.predict(uid,'feature_names'))
