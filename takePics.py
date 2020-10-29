import cv2
import cv2 as cv
import numpy
import time
import glob
import os
import uuid



OUTPUT_IMG_SIZE = 200




def main():
	


	print("Please insert your name: ")
	name = input()
	
	if not os.path.exists('data/' + name + '/'):
		os.makedirs('data/' + name + '/')
	images = []
	labels = []
	num = 0
	
	labels=numpy.asarray(labels)
	
#	cv2.namedWindow("camera", 1)
	capture = cv2.VideoCapture(0)
	while True:
		num=num+1
		ret,frame = capture.read()

		#rects, img = detect(iplimg)
		#for i in xrange(0,len(rects)):
		#	face_i = img[rects[i][1]:rects[i][3],rects[i][0]:rects[i][2]] # think about have a retangle and two point for it
		#	cv2.imwrite("temp.jpg",face_i)
		#	face_i = cv2.imread("temp.jpg",0)
	#		face_resized=cv2.resize(face_i, (100, 200), 1.0, 1.0, cv2.INTER_CUBIC);
			
			#cv2.imwrite(exchange(prediction[0])+str(num)+".jpg",face_resized)
	#		string = "here is "+exchange(prediction[0])
	#		fx = rects[i][0] -10
	#		fy = rects[i][1] -10
	#		cv2.putText(img,string,(fx,fy),cv2.FONT_HERSHEY_PLAIN, 1.0, cv2.cv.CV_RGB(0,255,0), 2)
	#	img = numpy.asarray(iplimg[:,:])
		cv2.imwrite("data/"+name+"/pic"+str(num)+".jpg",frame)
		#box(rects, img)
	#	gray = cv2.cvtColor(frame)
		cv2.imshow('frame', frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

		time.sleep(.3)
		if num == 50:
			break
	



	faceFiles = glob.glob('data/'+name+'/*.jpg')
	getFaces(faceFiles,name)




def getFaces(images,name):
	
	if not os.path.exists('faceData/' + name + '/'):
		os.makedirs('faceData/' + name + '/')
	for im in images:
		image = cv2.imread(im)
		# Load the OpenCV classifier to detect faces
		faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

		# Detect faces in the image
		faces = faceCascade.detectMultiScale(
			image,
			scaleFactor=1.2,
			minNeighbors=5,
			minSize=(50, 50)
		)

		# The faces variable now contains an array of Nx4 elements where N is the number faces detected

		print("Found", len(faces), "faces")


		for (x, y, w, h) in faces:
			img1 = image[y:y+h, x:x+w]

			#print ('Cropped image size is: ' + str(img1.shape))
			r = OUTPUT_IMG_SIZE / img1.shape[1]
			img2 = cv2.resize(img1, (OUTPUT_IMG_SIZE, OUTPUT_IMG_SIZE), interpolation = cv2.INTER_AREA)

			#generate a unique filename
			fname = "faceData/" + name + '/' + str(uuid.uuid4()) + ".png"



			print("Saving face:", fname)
			cv2.imwrite(fname, img2)



def detect(iplimg):
    cascade = cv2.CascadeClassifier("./haarcascade_frontalface_alt.xml")
    img = numpy.asarray(iplimg[:,:])
    rects = cascade.detectMultiScale(img, 1.3, 4, cv2.cv.CV_HAAR_SCALE_IMAGE, (20,20))

    if len(rects) == 0:
        return [], img
    rects[:, 2:] += rects[:, :2]
    return rects, img

def box(rects, img):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), (127, 255, 0), 2)
    return img


main()







