#Resize image 
import cv2
import subprocess
import glob
import os
import re
import numpy as np
from garbageImageDetector import isGarbageImage
from pylab import array, plot, show, axis, arange, figure, uint8 

def resizeImage(img):

	shape = img.shape
	rows = shape[0];
	cols = shape[1];

	if rows > cols:
		scale = 1000.0/rows
	else: 
		scale = 1000.0/cols

	img = cv2.resize(img,None,fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)

	return img

#morphological operations 
def morphological(img):
	kernel = np.ones((3,3),np.uint8)
	erosion = cv2.erode(img,kernel,iterations = 1)
	kernel = np.ones((2,2),np.uint8)
	dilation = cv2.dilate(erosion,kernel,iterations = 1)
	return dilation

#clustering 
def cluster(img):
	Z = img.reshape((-1,3))

	# convert to np.float32
	Z = np.float32(Z)

	# define criteria, number of clusters(K) and apply kmeans()
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	K = 8
	ret,label,center=cv2.kmeans(Z,K,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
	
	# Now convert back into uint8, and make original image
	center = np.uint8(center)
	res = center[label.flatten()]
	res2 = res.reshape((img.shape))
	return res2

#enhancing 
def sharpenImage(img):
	 blur = cv2.bilateralFilter(img,9,75,75)
	 weighted=cv2.addWeighted(img, 1.5, blur, -0.5, 0)
	 return weighted

def enhance(image):
	maxIntensity = 255.0 # depends on dtype of image data
	x = arange(maxIntensity) 

	# Parameters for manipulating image data
	phi = 1
	theta = 1

	# Increase intensity such that
	# dark pixels become much brighter, 
	# bright pixels become slightly bright
	newImage0 = (maxIntensity/phi)*(image/(maxIntensity/theta))**0.5
	newImage0 = array(newImage0,dtype=uint8)

	y = (maxIntensity/phi)*(x/(maxIntensity/theta))**0.5

	# Decrease intensity such that
	# dark pixels become much darker, 
	# bright pixels become slightly dark 
	newImage1 = (maxIntensity/phi)*(image/(maxIntensity/theta))**2
	newImage1 = array(newImage1,dtype=uint8)
	
	return newImage1

#Otsu's binarization 
def otsusBinarization(img):

	img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	# blur = cv2.bilateralFilter(img,9,75,75)
   	ret,img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	return img

#Otsu's binarization 
def iterativeThresholding(img):

	gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	threshold = 0
	new_threshold = 100
	i = 0

	while not new_threshold <= threshold + 0.001 and new_threshold >= threshold - 0.001:
		print i, new_threshold
		i += 1

		threshold = new_threshold
		th,th_img = cv2.threshold(gray_img,threshold,255,cv2.THRESH_BINARY)
		
		count_back = 0
		avg_back = 0 
		count_fore = 0
		avg_fore = 0

		for r,row in enumerate(th_img):
			for p,pixel in enumerate(row):
				if pixel == 255:
					avg_back += gray_img[r][p]
					count_back += 1
				else:
					avg_fore += gray_img[r][p]
					count_fore += 1

		avg_back = float(avg_back)/count_back
		# print count_back, avg_back
		avg_fore = float(avg_fore)/count_fore
		# print count_fore, avg_fore

		new_threshold = (avg_back + avg_fore)/2

	return th_img

def makeEnlarged(page, ENLR_THRESH):
	# pattern = re.compile('[0-9]+')
	# page, mc_num = pattern.findall(inImg)
	# page = int(page)

	print '=============================='
	print page
	print 'convert -density 300 -trim output_%02d.pdf -quality 100 -sharpen 0x1.0 raw_%02d.jpg' % ( page , page )
	print '=============================='

	subprocess.call('convert -density 300 -trim output_%02d.pdf -quality 100 -sharpen 0x1.0 raw_%02d.jpg' % ( page , page ), shell=True)
	# subprocess.call(["./multicrop", "raw_%02d.jpg" % page, "multicrop-output/new_raw_mc_%02d.ppm" % page])
	# subprocess.call(["./multicrop", "raw_%02d.jpg" % page, "-e", "100" ,"multicrop-ext-output/new_raw_mc_%02d.ppm" % page])

	subprocess.call("bash multicrop -u 3 " + ("raw_%02d.jpg " % page) + ("multicrop-output/new_raw_mc_%02d.ppm " % page), shell=True)
	subprocess.call("bash multicrop -e 100 -u 3 " + ("raw_%02d.jpg " % page) + ("multicrop-ext-output/new_raw_mc_%02d.ppm " % page), shell=True)
		

	new_imgs = glob.glob1('multicrop-ext-output/',"new_raw_mc_%02d*.ppm" % page)
	print '##########################################'
	print new_imgs
	print '##########################################'

	for i in new_imgs:
		print i
		img = cv2.imread('multicrop-ext-output/'+i)
		img_2 = cv2.imread('multicrop-output/'+i)
		height, width = img_2.shape[:2]
		if isGarbageImage(img):
			os.remove("multicrop-output/"+i)
			os.remove("multicrop-ext-output/"+i)
		elif max(height, width) > 2*ENLR_THRESH + 50:
			raw_input("About to delete %s ..." % i)
			os.remove("multicrop-output/"+i)
			os.remove("multicrop-ext-output/"+i)
		else:
			pass
			# os.rename('multicrop-output/'+i , 'multicrop-output/'+i.replace('new_', '', 1))
			# os.rename('multicrop-ext-output/'+i , 'multicrop-ext-output/'+i.replace('tmp_', '', 1))

	os.remove('raw_%02d.jpg' % page )



def genImages(inPdf, ENLR_THRESH=600):
	subprocess.call('pdftk '+ inPdf + ' burst output output_%02d.pdf', shell=True)
	subprocess.call('pdftoppm '+ inPdf + ' output ', shell=True)
	numPages = len(glob.glob1('.',"output_*.pdf"))
	
	for i in range(1, numPages+1):
		# subprocess.call('convert -density 150 -trim output_%02d.pdf -quality 100 -sharpen 0x1.0 raw_%02d.ppm' % ( i , i ), shell=True)
		print "run1"
		subprocess.call("bash multicrop -u 3 " + ("output-%d.ppm " % i) + ("multicrop-output/tmp_raw_mc_%02d.ppm " % i), shell=True)
		print "run2"
		subprocess.call("bash multicrop " +  "-e 100 -u 3 " + ("output-%d.ppm " % i) + ("multicrop-ext-output/tmp_raw_mc_%02d.ppm " % i), shell=True)
		
		mc_imgs = glob.glob1('multicrop-ext-output/',"tmp_raw_mc_%02d*.ppm" % i)

		print mc_imgs
		
		fin_files = []

		for currImg  in mc_imgs:
			img = cv2.imread('multicrop-ext-output/'+currImg)
			if isGarbageImage(img):
				print "is garbage! " + currImg
				os.remove('multicrop-ext-output/'+currImg)
				os.remove('multicrop-output/'+currImg)
			else:
				fin_files.append(currImg)

		raw_input("Phase 1")
		# raw_input("Phase 2")
		flag = False
		for croppedImg in fin_files:
			img = cv2.imread('multicrop-output/'+croppedImg)
			height, width = img.shape[:2]
			if max(height, width) <= ENLR_THRESH:
				os.remove('multicrop-output/'+croppedImg)
				os.remove('multicrop-ext-output/'+croppedImg)
				flag = True
		
		if flag:
			makeEnlarged(i, ENLR_THRESH)


		# x_imgs = glob.glob1('multicrop-ext-output/',"tmp_raw_mc_%02d*.ppm" % i)

		# for ximg in x_imgs:
		# 	os.rename('multicrop-output/'+ximg , 'multicrop-output/'+ximg.replace('tmp_', '', 1))
		# 	os.rename('multicrop-ext-output/'+ximg , 'multicrop-ext-output/'+ximg.replace('tmp_', '', 1))

		raw_input("Press Enter to continue...")


if __name__=="__main__":
	img_inp = cv2.imread("graphimage.png")
	img=resizeImage(img_inp)
	img1=sharpenImage(img)
	cv2.imwrite("sharpened.tiff",img1)
	img2=cluster(img1)
	cv2.imwrite("clustered.tiff",img2)
	out = otsusBinarization(img2)
	out1=enhance(out)
	out2=morphological(out1)
	cv2.imwrite("out.tiff",out2)
