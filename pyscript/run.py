import os

workDir = './'
outputDir = 'matlabcode/imagenet/'
probLayer = 'prob'
weights = 'zoo/caffe_reference_imagenet.caffemodel'
proto = 'zoo/caffe_reference_imagenet.prototxt'
probeBatch = '64'
mean = '109.5388 118.6897 124.6901'
colorDepth = '3'
dataset = 'dataset/broden1_227/'
dataSize = '10000'
gpu = '1'

os.system('python src/netprobe.py' + \
    ' --directory ' + workDir + '/' + outputDir +\
    ' --blobs ' + probLayer + \
    ' --weights ' + weights + \
    ' --definition ' + proto + \
    ' --batch_size ' + probeBatch + \
    ' --mean ' + mean + \
    ' --colordepth ' + colorDepth + \
    ' --dataset ' + dataset + \
    ' --limit ' + dataSize + \
    ' --gpu ' + gpu)

# os.system('python src/quantprobe.py' + \
#     '--directory ' + workDir + '/' + dir + \
#     ' --blobs ' + probLayer)
