from mtcnn.mtcnn_detector import MtcnnDetector
import mxnet as mx
import cv2
PREPROCESS='mtcnn'

def main(args):

	if 1:
		cap = cv2.VideoCapture(0) 
		while(1): # get a frame ret
			ret,img = cap.read() # show a frame 
			cv2.imshow("capture", img) 
			if cv2.waitKey(1) & 0xFF == ord('1'):	
				break 
		cap.release()
	if PREPROCESS=='mtcnn':
		img = cv2.resize(img, (320,180))
		detector = MtcnnDetector(model_folder='mtcnn/model', ctx=mx.gpu(0), num_worker = 1 , accurate_landmark = True)
		results = detector.detect_face(img)
		total_boxes = results[0]
		points = results[1]
		chips = detector.extract_image_chips(img, points, 112, 0.37)
		img=chips[0]
	cv2.imshow('face',img)
	cv2.waitKey(0)
main(1)