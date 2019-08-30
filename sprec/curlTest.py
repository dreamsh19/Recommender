
import random
import os
import recommender as rc

users=2100

testCase=10

for i in range(testCase):

	uid_random=random.randint(1,users)
	X=[[uid_random]]
	
	print('USER ID [%d]' %uid_random)
	os.system('curl -g http://localhost:5000/predict -d  \'json={\"data\": {\"names\": [\"inputName\"], \"ndarray\": [['+str(uid_random)+']]}}\'')
	
