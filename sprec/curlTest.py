
import random
import os
import recommender as rc

users=1892

testCase=10

for i in range(testCase):
	X=[[]]

	uid_random=random.randint(1,users)
	X[0].append(uid_random)
	
	print('USER ID [%d]' %uid_random)
	os.system('curl -g http://localhost:5000/predict -d  \'json={\"data\": {\"names\": [\"inputName\"], \"ndarray\": [['+str(uid_random)+']]}}\'')


#print('uid[%d]' %uid_random)

	