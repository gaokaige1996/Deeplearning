file1 = open('valid_image_paths.txt')
file2 = open('valid_labeled_studies.txt')
path = []
for line in file1:
    line = line.strip('\n')
    path.append(line)
with open('lable image_valid.txt','w') as f:
    n = 0
    for line1 in file2.readlines():
        line1 = line1.strip('\n')
        l = line1.split('\t')[0]
        #print(l)
        lable = line1.split('\t')[1]
        for i in path:
            if l in i:
               f.write(i + '\t'+lable + '\n')
            else:
               pass


