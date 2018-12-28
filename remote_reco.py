from mtcnn.mtcnn_detector import MtcnnDetector
import numpy as np
import cv2
import mxnet as mx
import sys
import socket
import struct
from feature_extractor import extractor as feature_extractor
from database import database_op
import os
import threading
class recogize:
	def __init__(self):
		self.detector = MtcnnDetector(model_folder='mtcnn/model', ctx=mx.gpu(0), num_worker = 1 , accurate_landmark = True)
		img=cv2.imread('pic/t1,jpg')
		self.detector.detect_face(img)
		print ('detector prepare succ!')
		self.extractor=feature_extractor('models/se-softmax/test17,114')
		print ('extractor prepare succ!')
		self.np_path='dataset/embs.bin'
		self.dop=database_op()
		try: 
			sock=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
			print("create socket succ!")
			sock.bind(("192.168.70.161",8995))
			sock.listen(5)
			self.sock=sock
		except BaseException,e:
			print e
			print("init socket error!")
			sys.exit(0)

	def run(self):
		while 1:
			self.names=self.dop.select_all()
			self.embs=np.fromfile(self.np_path,dtype=np.float32)
			if self.embs.shape[0]>0:
				self.embs=self.embs.reshape((-1,512))
			assert self.embs.shape[0]==len(self.names)
			print 'wait for client...'
			timer = threading.Timer(120.0, self.release)
			timer.start()
			self.conn,addr=self.sock.accept()
			print("get client") 
			print(addr)
			timer.cancel() 		 
			self.conn.settimeout(30)
			mode=self.recvlen(self.conn)
			if (mode==111):
				print 'mode:recognize'
				self.func=self.loop_rec
			elif(mode==222):
				print 'mode:insert'
				self.func=self.insert_database
			self.func()


	def loop_rec(self):
		conn=self.conn
		self.conn.settimeout(5)
		while 1:
			final_result=''
			try:
				l=self.recvlen(conn)
				data=self.recvdata(conn,l)
			except BaseException,e:
				print e
				cv2.destroyAllWindows()
				conn.close()
				return				
			img=self.decode(data)
			img=np.fliplr(np.rot90(img))
			cv2.imshow('aa',img)
			cv2.waitKey(60)
			results=self.detector.detect_face(img)
			if results!=None:
				points = results[1]
				points=np.expand_dims(points[0],0)
				chips=self.detector.extract_image_chips(img, points, 112, 0.37)
				data=np.transpose(chips[0],(2,0,1))
				data=np.expand_dims(data,0)
				embedding=self.extractor.extract_feature(data)
				embs=self.embs.T
				res=np.dot(embedding,embs)[0]
				for i in range(len(res)):
					print '%s:%f'%(self.names[i],res[i]),
				print ''
				most_possible_identity=np.argmax(res)
				if res[most_possible_identity]>=0.55:
					final_result=self.names[most_possible_identity]
					print 'result:%s'%final_result
			meg='0'*20
			if len(final_result):
				meg=meg.replace('0','1',1)
				meg=meg.replace('0',str(len(final_result)),1)
				meg=meg.replace('0'*len(final_result),final_result,1)
				conn.sendall(meg)
				break
			else:
				conn.sendall(meg)
		cv2.destroyAllWindows()
		conn.close()

	def insert_database(self):
		conn=self.conn
		l=self.recvlen(conn)
		if l==0:
			return
		data=self.recvdata(conn,l)
		name=self.recvname(conn)
		print name
		img=self.decode(data)
		img=np.fliplr(np.rot90(img))
		results=self.detector.detect_face(img)
		num_emb=self.embs.shape[0]
		if results!=None:
			points = results[1]
			points=np.expand_dims(points[0],0)
			chips=self.detector.extract_image_chips(img, points, 112, 0.37)[0]
			backup=os.path.join('/home/leo/unlock_by_face/pic/database/',name+'.jpg')
			print backup
			print chips.shape
			cv2.imwrite(backup,chips)
			data=np.transpose(chips,(2,0,1))
			data=np.expand_dims(data,0)
			embedding=self.extractor.extract_feature(data)
			idx=self.dop.insert(name)
			print idx
			if idx==num_emb:
				if num_emb==0:
					embs=np.expand_dims(embedding,0)
				else:
					self.embs=np.append(self.embs,embedding,axis=0)
				self.embs.tofile(self.np_path)
				print 'Save Successfully!!'
			elif idx<num_emb:
				self.embs[idx]=embedding
				self.embs.tofile(self.np_path)
				print 'Replace Successfully!!'
			else:
				print 'Dataset is not sync with bin file, something must be wrong.'
		conn.close()		

	def release(self):
		self.dop.__del__()
		self.sock.close()
		os._exit(0)

	def recvlen(self,conn):
		bytes=conn.recv(4)
		l=struct.unpack('i',bytes)
		return l[0]

	def recvdata(self,conn,l):
		data=''
		while len(data)<l:
			data+=conn.recv(min(1024,l-len(data)))
		return data

	def recvname(self,conn):
		bytes=conn.recv(20)
		l=struct.unpack('b',bytes[0])[0]
		name=bytes[1:1+l].encode('UTF-8')
		return name.upper()


	def decode(self,data):
		return cv2.imdecode(np.fromstring(data, np.uint8), cv2.IMREAD_COLOR )


rec=recogize()
rec.run()



