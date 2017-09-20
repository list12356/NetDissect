import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread
import time

inputFileName = './matlabcode/broden_labled.csv'
inputImageFile = './dataset/broden1_227/index.csv'
inputImageDir = './dataset/broden1_227/images/'


# Construct the image file dictionary
imageDict = []
with open(inputImageFile) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        imageDict.append(row['image'])

# Construct the test class label index file
objectNum = ['131', '93', '105', '123']; # 105: cat 131: house 93: dog 123: boat
objectList = 'house-dog-cat-boat'
outputFileName = './matlabcode/' + objectList + '.csv'
outputList=[[]]
lineNum = 0;
with open(inputFileName) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        # Index strat from 1, since matalb index start from 1
        lineNum = lineNum + 1
        tmpList = row['object'][1:-1].split(' ')
        if tmpList != '':
            for object in objectNum:
                if object in tmpList:
                    # plt.imshow(imread(inputImageDir + imageDict[lineNum - 1]))
                    # plt.show()
                    # plt.pause(0.5)
                    # plt.close('all')
                    outputList.append([lineNum, object])

with open(outputFileName, 'w') as csvfile:
    fieldnames=['line_num', 'object']
    writer = csv.DictWriter(csvfile, fieldnames = fieldnames)

    writer.writeheader()
    for line in outputList:
        if len(line) == 2:
            writer.writerow({'line_num': line[0], 'object': line[1]})

for objectName in objectList.split('-'):
    with open(objectName + '.imglist', 'w') as file:
        for line in outputList:
            if len(line) == 2 and line[1] == '123':
                file.write('./image_broden/' + '/' + imageDict[int(line[0]) - 1]+'\n')
            
