import caffe
import numpy as np
import expdir
import loadseg


caffe.set_mode_cpu()

rootDir = "./"

model_def = rootDir + '/zoo/caffe_reference_places365.prototxt'
model_weights = rootDir + '/zoo/caffe_reference_places365.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)
with open(rootDir + '/dataset/testlist.txt', 'r') as testFile:
    testImageList = testFile.read().splitlines()

dataSize = len(testImageList)
batch_size = 50
blobs=['conv5']
output = {}

ed = expdir.ExperimentDirectory(rootDir + '/pythonScriptTest/')
ed.ensure_dir()
for blob in blobs:
    shape = (dataSize, ) + net.blobs[blob].data.shape[1:]
    output[blob] = ed.open_mmap(blob=blob, part='tes-nmf', mode='w+', shape=shape)

pf = loadseg.SegmentationPrefetcher(data, categories=['image'],
        split=None, once=True, batch_size=batch_size, ahead=ahead)


mean = np.array([109.5388, 118.6897, 124.6901], dtype=np.float32)
input_blob = net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs[input_blob].data.shape})
transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mean)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

for line in testImageList:
    image = caffe.io.load_image(rootDir + line)
    transformed_image = transformer.preprocess('data', image)
    net.blobs[input_blob].data[...] = transformed_image
    result = net.forward(blobs=blobs)
    for blob in blobs:
        output[blob] = result[blob]
        ed.finish_mmap(output[blob])