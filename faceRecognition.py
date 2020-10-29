
from collections import Counter

import cv2
import glob
import scipy.fftpack
import numpy as np
from scipy.spatial import distance

OUTPUT_IMG_SIZE = 200

#-----------------------------------
def euclideannn(vector1, vector2):
    ''' use scipy to calculate the euclidean distance. '''
    dist = distance.euclidean(vector1, vector2)
    return dist


def zigagElements(mat, row, col) :

    i = 0
    j = 0
    outt = []

    while (i < row) :


        outt.append(mat[i,j])

        if i == row - 1 :
            i = j + 1
            j = col - 1
        elif j == 0 :
            j = i + 1
            i = 0


        else :
            j = j-1
            i = i+1

    return outt
#-------------------------------------





def getFaces(image):
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    faces = faceCascade.detectMultiScale(
        image,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(50, 50)
    )

    # The faces variable now contains an array of Nx4 elements where N is the number faces detected

    print("Found", len(faces), "faces")
    if len(faces) == 0 :
        return 0

    for (x, y, w, h) in faces:
        img1 = image[y:y + h, x:x + w]

        # print ('Cropped image size is: ' + str(img1.shape))
        r = OUTPUT_IMG_SIZE / img1.shape[1]
        img2 = cv2.resize(img1, (OUTPUT_IMG_SIZE, OUTPUT_IMG_SIZE), interpolation=cv2.INTER_AREA)

        return img2
        # # generate a unique filename
        # fname = "faceData/" + name + '/' + str(uuid.uuid4()) + ".png"
        #
        # print("Saving face:", fname)
        # cv2.imwrite(fname, img2)

#---------------------------------------------------------------------------
def __get_images_and_labels(files):
	print("Loading faces for training")
	
	images = []
	labels = []

	c = 0
	for f in files:
		print(c, "-", f)
		#cv2.imshow("Adding faces to traning set...", cv2.imread(f))
		#cv2.waitKey(50)
		images.append(__prepare_image(f))
		labels.append(c)
		c = c+1
	
	cv2.destroyAllWindows()
	
	return images, labels


# ---------------------------------------------------------------------------
def DctRecognize( dctss , testimg ):
	# print("Loading faces for training")

	eclideans = []
	labels = []


	imftest = np.float32(testimg) / 255.0  # float conversion/scale
	dsttest = cv2.dct(imftest)  # the dct
	testimg = dsttest * 255.0  # convert back

	finouttest = zigagElements(testimg, np.size(testimg, 1), np.size(testimg, 1))
	low = finouttest[3:2003]
	mid = finouttest[13333:15333]
	high = finouttest[26666:28666]
	testfs = low + mid + high
	TrainDcts = []
	c = 0
	for f in range(len(dctss)):
		# print(c, "-", f)
		# cv2.imshow("Adding faces to traning set...", cv2.imread(f))
		# cv2.waitKey(50)
		# img  = __prepare_image(f)
        #
		# # img = cv2.imread('a.png', 0)  # 1 chan, grayscale!
		# imf = np.float32(img) / 255.0  # float conversion/scale
		# dst = cv2.dct(imf)  # the dct
		# img = dst * 255.0  # convert back

		# finoutt = zigagElements(img, np.size(img, 1), np.size(img, 1))
        #
		# low = finoutt[3:2003]
        #
		# mid = finoutt[13333:15333]
        #
		# high = finoutt[26666:28666]
        #
		# ariafs = low + mid + high
        #
		# TrainDcts.append(ariafs)

		eclideans.append(euclideannn(testfs,dctss[f][:]))
		labels.append(c)
		c = c + 1

	cv2.destroyAllWindows()

	return eclideans

# ---------------------------------------------------------------------------
def GetDcts(files):

	TrainDcts = []
	c = 0

	for f in files:
		print(c, "-", f)
		# cv2.imshow("Adding faces to traning set...", cv2.imread(f))
		# cv2.waitKey(50)
		img = __prepare_image(f)

		# img = cv2.imread('a.png', 0)  # 1 chan, grayscale!
		imf = np.float32(img) / 255.0  # float conversion/scale
		dst = cv2.dct(imf)  # the dct
		img = dst * 255.0  # convert back

		finoutt = zigagElements(img, np.size(img, 1), np.size(img, 1))

		low = finoutt[3:2003]

		mid = finoutt[13333:15333]

		high = finoutt[26666:28666]

		ariafs = low + mid + high

		TrainDcts.append(ariafs)

	return TrainDcts
# ---------------------------------------------------------------------------
def getMeans(persons):

	meanss = []
	c = 0

	for f in persons:
		eachperson = glob.glob( f + '/*.png')
		num = 0
		meann2 = np.zeros((200,200))
		for e in eachperson :
			img = __prepare_image(e)
			# imgg = np.asmatrix(img)
			num = num + 1
			meann2 = np.matrix(img) + meann2
		print(num)
		meann2 = meann2 / num
		# meann.flatten()

		meanss.append(meann2)


	return meanss






#---------------------------------------------------------------------------
# Load the image file, convert	to greyscale, normalize brightness and
# return the image
def __prepare_image(filename):
	img = cv2.imread(filename)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = cv2.equalizeHist(img)
	return img


def whoIs(fName):
	print(fName, end=' ')
	
	img = __prepare_image(fName)

	collector = cv2.face.StandardCollector_create()
	recognizer.predict_collect(img, collector)
	dist = collector.getMinDist()
	nbr_predicted = collector.getMinLabel()


	print(">>>", faceFiles[nbr_predicted], " (dist="+str(int(dist))+")")


def NNMean(MeanofPs, faceimg):
	eccs = []
	for f in MeanofPs :
		eccs.append(euclideannn(f.flatten(), faceimg.flatten()))
	return eccs







recognizer = cv2.face.createFisherFaceRecognizer()
recognizer2 = cv2.face.createEigenFaceRecognizer()
recognizer3 = cv2.face.createLBPHFaceRecognizer()

faceFiles = glob.glob('faceData/*/*.png')
images, labels = __get_images_and_labels(faceFiles)

dctss = GetDcts(faceFiles)



persons = glob.glob('faceData/*')

MeanofPs = getMeans(persons)

# eachperson = glob.glob(faceFiles[1] + '/*.png')

print(len(dctss))

recognizer.train(images, np.array(labels))
recognizer2.train(images, np.array(labels))
recognizer3.train(images, np.array(labels))

# Load the sample image
# whoIs('data/Agnetha1.png')

#------------------------------------------------
capture = cv2.VideoCapture(0)
num = 0


persons2 = glob.glob('faceData/*/')

numofiters = 0

while True:

	names = []

	num = num + 1
	ret, img = capture.read()
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = cv2.equalizeHist(img)
	collector = cv2.face.StandardCollector_create()
	faceimg = getFaces(img)
	if isinstance( faceimg, int ):
		continue
	#------------1------------
	recognizer.predict_collect(faceimg, collector)
	dist = collector.getMinDist()
	nbr_predicted = collector.getMinLabel()
	name = faceFiles[nbr_predicted].partition('\\')[-1].rpartition('\\')[0]
	names.append(name)
	# ------------2------------
	recognizer2.predict_collect(faceimg, collector)
	dist = collector.getMinDist()
	nbr_predicted = collector.getMinLabel()
	name = faceFiles[nbr_predicted].partition('\\')[-1].rpartition('\\')[0]
	names.append(name)
	#-------------3------------
	recognizer3.predict_collect(faceimg, collector)
	dist = collector.getMinDist()
	nbr_predicted = collector.getMinLabel()
	name = faceFiles[nbr_predicted].partition('\\')[-1].rpartition('\\')[0]
	names.append(name)

	# -----------DCT-----------
	eucs = DctRecognize(dctss, faceimg)
	minind = np.argmin(eucs)
	name = faceFiles[minind].partition('\\')[-1].rpartition('\\')[0]
	names.append(name)
	# -----------Nearest Neighbor Mean-----------

	nneccs = NNMean(MeanofPs , faceimg)
	name = persons2[np.argmin(nneccs)].partition('\\')[-1].rpartition('\\')[0]
	names.append(name)

	print(names)
	maxp = Counter(names)
	maxppandnum = maxp.most_common(1)

	if maxppandnum[0][1] > 2 :
		numofiters = numofiters + 1
	if numofiters > 29 :
		print('This Person is : ')
		print(maxppandnum[0][0])
		print('Fin.')
		break;




	# print(">>>", name, " (dist="+str(int(dist))+")")


# Wait for any key before exiting
cv2.waitKey(0)



