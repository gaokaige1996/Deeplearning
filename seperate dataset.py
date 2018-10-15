file = open('lable image_valid.txt')
catl = []
shoulder = []
wrist = []
hand = []
finger = []
elbow = []
humerus = []
forearm = []
for line in file.readlines():
    #for line1 in file2.readlines():
        #line1 = line1.strip('\n')
        #l = line1.split('\t')[0]
    l = line.strip('\n')
    category = l.split('/')[2]
    if category == 'XR_WRIST':
        wrist.append(l)
    elif category == 'XR_HAND':
        hand.append(l)
    elif category == 'XR_FINGER':
        finger.append(l)
    elif category == 'XR_SHOULDER':
        shoulder.append(l)
    elif category == 'XR_ELBOW':
        elbow.append(l)
    elif category == 'XR_HUMERUS':
        humerus.append(l)
    elif category == 'XR_FOREARM':
        forearm.append(l)
    #catl.append(category)
#print(set(catl))
l = [wrist,hand,finger,shoulder,elbow,humerus,forearm]
name = ['wrist','hand','finger','shoulder','elbow','humerus','forearm']
for i in range(len(l)):
    with open(name[i]+'_valid.txt','w') as f:
        for m in l[i]:
            f.write(m + '\n')