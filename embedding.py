import cv2
import argparse
import numpy as np
from feature_extractor import extractor as feature_extractor
from SSH.ssh_detector import SSHDetector
import json
from database import database_op
from align_dlib import AlignDlib
from mtcnn.mtcnn_detector import MtcnnDetector
import mxnet as mx
import random

PREPROCESS='mtcnn'

def main(args):
	if args.fromcamera==0:
		img = cv2.imread(args.path)
	else:
		cap = cv2.VideoCapture(0) 
		while(1): # get a frame ret
			ret,img = cap.read() # show a frame 
			cv2.imshow("capture", img) 
			if cv2.waitKey(1) & 0xFF == ord('1'):	
				break 
		cap.release()
	if PREPROCESS=='SSH':
		#resize picture
		detector = SSHDetector('models/detector/e2ef,0')
		im_shape = img.shape
		print(im_shape)
		scales = [300, 400]
		target_size = scales[0]
		max_size = scales[1]
		im_size_min = np.min(im_shape[0:2])
		im_size_max = np.max(im_shape[0:2])
		if im_size_min>target_size or im_size_max>max_size:
			im_scale = float(target_size) / float(im_size_min)
			if np.round(im_scale * im_size_max) > max_size:
					im_scale = float(max_size) / float(im_size_max)
			img = cv2.resize(img, None, None, fx=im_scale, fy=im_scale)
		print('resize to', img.shape)
		faces = detector.detect(img, threshold=0.5)
		#resize face
		if not faces.shape[0]>=1:
			print 'no face found'
			return 0
		face=map(int,faces[0])
		width=face[2]-face[0]+1
		height=face[3]-face[1]+1
		sub=width-height
		if sub>0:
			face[3]+=sub/2
			face[3]=min(img.shape[0]-1,face[3])
			face[1]-=sub/2
			face[1]=max(0,face[1])
		else:
			face[2]-=sub/2
			face[2]=min(img.shape[1]-1,face[2])
			face[0]+=sub/2
			face[0]=max(0,face[0])
		img=img[face[1]:face[3]+1,face[0]:face[2]+1,:]
	elif PREPROCESS=='Dlib':
		aligner=AlignDlib('models/shape_predictor_68_face_landmarks.dat')
		img=aligner.align(112,img)
	elif PREPROCESS=='mtcnn':
		img = cv2.resize(img, (320,180))
		detector = MtcnnDetector(model_folder='mtcnn/model', ctx=mx.gpu(0), num_worker = 1 , accurate_landmark = True)
		results = detector.detect_face(img)
		total_boxes = results[0]
		points = results[1]
		chips = detector.extract_image_chips(img, points, 112, 0.37)
		img=chips[0]
	cv2.imshow('face',img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()	
	data=np.transpose(img,(2,0,1))
	data=np.expand_dims(data,0)
	#extract embeddings
	extractor=feature_extractor('models/model-r34-amf/model,0')
	embedding=extractor.extract_feature(data)
	#read dataset
	embs=np.fromfile(args.np_path,dtype=np.float32)
	if embs.shape[0]>0:
		embs=embs.reshape((-1,512))
	num_emb=embs.shape[0]
	dop=database_op()
	#insert or replace
	if args.mode==0:
		print 'Enter The Name of That Person:'
		name=raw_input()
		backup=args.backup_path+name+'.jpg'
		cv2.imwrite(backup,img)
		idx=dop.insert(name)
		print idx
		if idx==num_emb:
			if num_emb==0:
				embs=np.expand_dims(embedding,0)
			else:
				embs=np.append(embs,embedding,axis=0)
			embs.tofile(args.np_path)
			print 'Save Successfully!!'
		elif idx<num_emb:
			embs[idx]=embedding
			embs.tofile(args.np_path)
			print 'Replace Successfully!!'
		else:
			print 'Dataset is not sync with bin file, something must be wrong.'
	elif args.mode==1:
		backup=args.testsave_path+str(random.randint(0,999999))+'.jpg'
		cv2.imwrite(backup,img)
		names=dop.select_all()
		embs=embs.T
		res=np.dot(embedding,embs)[0]
		print res
		print names
		most_possible_identity=np.argmax(res)
		if res[most_possible_identity]>=0.55:
			print names[most_possible_identity]
		else:
			print 'No Match in Database'	



		

def parse_args():
  parser = argparse.ArgumentParser(description='Train face network')
  # general
  parser.add_argument('--mode', default=0, type=int, help='0:save people to dateset,1:face recognize')
  parser.add_argument('--path', default='pic/t16.jpg', help='picture path')
  parser.add_argument('--backup_path', default='pic/database/', help='picture path')
  parser.add_argument('--testsave_path', default='pic/testcase/', help='picture path')
  parser.add_argument('--fromcamera', default=0, type=int, help='picture from camera')
  parser.add_argument('--np_path',default='dataset/embs.bin',help='')
  args = parser.parse_args()
  return args


if __name__ == '__main__':
    main(parse_args())
