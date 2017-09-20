import loadseg
import numpy as np
import csv

dataset = "./dataset/broden1_227/"
outputFileName = './matlabcode/broden_labled.csv'
lineNum = 0;
# Load the dataset
data = loadseg.SegmentationData(dataset)
categories = data.category_names()
print categories
fieldnames=['line_num', 'color', 'object', 'part', 'material', 'scene', 'texture']
print fieldnames

pf = loadseg.SegmentationPrefetcher(data, categories=categories, once=True,
        batch_size= 64, thread=False)

with open(outputFileName, 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
    writer.writeheader()

    for batch in pf.batches():
        for rec in batch:
            lineNum = lineNum + 1
            tmpLabel = [np.unique(rec[cat]) for cat in categories]
            writer.writerow({'line_num': lineNum, 'color': tmpLabel[0], 'object': tmpLabel[1], 
                'part': tmpLabel[2], 'material' : tmpLabel[3], 'scene' : tmpLabel[4], 'texture' : tmpLabel[5]})
        print lineNum
    