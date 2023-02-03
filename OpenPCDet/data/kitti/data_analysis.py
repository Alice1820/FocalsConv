#get file object reference to the file
file = open("/Users/alice/Documents/3DVision/devkit_object/mapping/train_mapping.txt", "r")

#read content of file to string
data = file.read()

dict = {}
#get number of occurrences of the substring in the string
for line in file:
    for word in line.split():
        # print (word)
        if word not in dict and word[-4:]=='sync':
            occurrences = data.count(word)
            dict[word] = occurrences
            # print('Number of occurrences of the word {}: {}'.format(word, occurrences))
print (dict)

import csv

with open('kitti_analysis.csv', 'w') as csv_file:  
    writer = csv.writer(csv_file)
    for key, value in dict.items():
       writer.writerow([key, value])