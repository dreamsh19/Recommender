import requests
import os
import math

DATA_DIRECTORY_PATH='./lastfm_data'
URL= 'http://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip'
ZIP_FILE_NAME='hetrec2011-lastfm-2k.zip'


def data_generate():

	r=requests.get(URL,allow_redirects=True)
	open(ZIP_FILE_NAME,'wb').write(r.content)

	os.system('rm -rf '+DATA_DIRECTORY_PATH[2:])
	os.system('mkdir -p '+DATA_DIRECTORY_PATH[2:])
	os.system('unzip '+ZIP_FILE_NAME+' -d '+DATA_DIRECTORY_PATH)
	os.system('rm '+ZIP_FILE_NAME)

	f=open(DATA_DIRECTORY_PATH + '/user_artists.dat','rt')
	fWrite=open(DATA_DIRECTORY_PATH +'/user_artists_log.dat','w+')

	f.readline()
	for line in f:
		tokens=line.split('\t')
		tokens[2]=str(math.log10(int(tokens[2][:-1])))
		writeLine=tokens[0]+'\t'+tokens[1]+'\t'+tokens[2]+'\r\n'
		fWrite.write(writeLine)
	f.close()
	fWrite.close()
	print('Data Generated')
