import requests
import os
import math

data_directory_path='./lastfm_data'

url= 'http://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip'
zipFileName='hetrec2011-lastfm-2k.zip'


def data_generate():

	r=requests.get(url,allow_redirects=True)
	open(zipFileName,'wb').write(r.content)

	os.system('rm -rf '+data_directory_path[2:])
	os.system('mkdir -p '+data_directory_path[2:])
	os.system('unzip '+zipFileName+' -d '+data_directory_path)
	os.system('rm '+zipFileName)

	f=open(data_directory_path + '/user_artists.dat','rt')
	fWrite=open(data_directory_path +'/user_artists_log.dat','w+')

	f.readline()
	for line in f:
		tokens=line.split('\t')
		tokens[2]=str(math.log10(int(tokens[2][:-1])))
		writeLine=tokens[0]+'\t'+tokens[1]+'\t'+tokens[2]+'\r\n'
		fWrite.write(writeLine)
	f.close()
	fWrite.close()
	print('Data Generated')
