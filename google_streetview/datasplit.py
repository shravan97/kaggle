import cv2
import numpy as np
import csv

def data():
	x = []
	for k in xrange(1,6284):
		img = cv2.imread('/home/shravan97/Downloads/trainResized/'+str(k)+'.Bmp' , 0)
		x.append(np.reshape(img,(400,1)))
	
	label_file = open('/home/shravan97/Downloads/trainLabels.csv' , 'r')
	l = csv.reader(label_file , delimiter=',')
	y=[]
	for j in l:
		if j[0]!='ID':
			y.append(j[1])	

	x , y= np.array(x),np.array(y)		

	return np.array(x.reshape(6283,400)) ,np.array(y.reshape(6283,1))			