import csv
inputFileName = './dataset/broden1_227/index.csv'
outputFileName = './matlabcode/broden_scene_labled.csv'
outputList=[[]]
lineNum = 0;
with open(inputFileName) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        lineNum = lineNum + 1
        if row['scene'] != '':
            outputList.append([lineNum, row['scene']])

with open(outputFileName, 'w') as csvfile:
    fieldnames=['line_num', 'scene']
    writer = csv.DictWriter(csvfile, fieldnames = fieldnames)

    writer.writeheader()
    for line in outputList:
        if len(line) == 2:
            writer.writerow({'line_num': line[0], 'scene': line[1]})



