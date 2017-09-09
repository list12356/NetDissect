import os

workDir = './'
dir = 'pythonScriptTest'
layers = 'conv5'
weights = 'zoo/caffe_reference_places365.caffemodel'
proto = 'zoo/caffe_reference_places365.prototxt'
probeBatch = '64'
mean = '109.5388 118.6897 124.6901'
colorDepth = '3'
dataset = 'dataset/broden1_227/'
gpu = 'true'

print 'Testing activations'
os.system('python src/netprobe.py' + \
    ' --directory ' + workDir + '/' + dir +\
    ' --blobs ' + layers + \
    ' --weights ' + weights + \
    ' --definition ' + proto + \
    ' --batch_size ' + probeBatch + \
    ' --mean ' + mean + \
    ' --colordepth ' + colorDepth + \
    ' --dataset ' + dataset + \
    ' --gpu ' + gpu)
