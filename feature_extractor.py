import mxnet as mx
import cv2
from sklearn import preprocessing

class extractor:
    def __init__(self,pretrained,ctx_id=0):
    	vec = pretrained.split(',')
    	sym, arg_params, aux_params = mx.model.load_checkpoint(vec[0], int(vec[1]))
        self.model = mx.mod.Module(
        context       = [mx.gpu(ctx_id)],
        symbol        = sym[0],
        label_names=[]
    	)
        self.model.bind(for_training=False, data_shapes=[('data', (1, 3, 112, 112))])
        self.model.set_params(arg_params=arg_params, aux_params=aux_params)


    	
    def extract_feature(self,img):
    	assert img.shape==(1,3,112,112)
    	data=mx.nd.array(img)
    	db = mx.io.DataBatch(data=(data,))
    	self.model.forward(db, is_train=False)
    	net_out = self.model.get_outputs()
    	embeddings = net_out[0].asnumpy()
    	embeddings = preprocessing.normalize(embeddings)
    	return embeddings


#pretrained='models/model-r34-amf/model,0'
#img_path='pic/WIN_20180403_09_31_48_Pro.jpg'

# img=cv2.imread(img_path,1)
# img=cv2.resize(img,(112,112))
# data=mx.nd.array(img)
# data=data.transpose((2,0,1))
# data=data.expand_dims(0)
# print data.shape
# print data.shape==(1,3,112,112)
# db = mx.io.DataBatch(data=(data,))

# ctx = [mx.gpu(0)]
# vec = pretrained.split(',')
# sym, arg_params, aux_params = mx.model.load_checkpoint(vec[0], int(vec[1]))
# model = mx.mod.Module(
#         context       = ctx,
#         symbol        = sym,
#         label_names=[]
#     )
# model.bind(for_training=False, data_shapes=[('data', (1, 3, 112, 112))])
# model.set_params(arg_params=arg_params, aux_params=aux_params)
# model.forward(db, is_train=False)
# net_out = model.get_outputs()
# embeddings = net_out[0].asnumpy()
# embeddings = preprocessing.normalize(embeddings)[0]
# print type(embeddings)


