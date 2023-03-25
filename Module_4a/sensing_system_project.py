import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
import time
import seaborn as sns

img_folder = '../data/'

# SIFT Timing
sift = cv.xfeatures2d.SIFT_create(300)
total_time=0
total_features=0

files = os.listdir(img_folder)
files = [file for file in files if file.find(".png") > 0]

for imgname in files:
    img = cv.imread(os.path.join(img_folder, imgname))
    start_time=time.time()
    kp,des = sift.detectAndCompute(img,None)
    total_time+=time.time()-start_time
    total_features+=np.array(kp).shape[0]
    sift_image = cv.drawKeypoints(img,kp, img, flags = cv.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
    #cv.imshow("Imshow", sift_image)
    #cv.waitKey(10000)
#cv.destroyAllWindows()
sift_time=total_time/2
sift_features = total_features/2
print("Average time for SIFT features: ",sift_time)
print("Average number of SIFT features: ",sift_features)

# ORB Timing
orb = cv.ORB_create(300)
total_time=0
total_features=0
for imgname in files:
    img = cv.imread(os.path.join(img_folder, imgname))
    start_time=time.time()
    kp,des = orb.detectAndCompute(img,None)
    total_time+=time.time()-start_time
    total_features+=np.array(kp).shape[0]
orb_time=total_time/2
orb_features = total_features/2

print("Average time for ORB features: ",orb_time)
print("Average number of ORB features: ",orb_features)

img2=cv.drawKeypoints(img,kp,img)
plt.imshow(img2)
plt.axis('off')

fig = plt.figure()
sns.set()
ax = fig.add_axes([0,0,1,1])
methods = ['SIFT',  'ORB']
times = [sift_time*1000, orb_time*1000]
ax.barh(methods,times,color=('green','orange'))
ax.set_ylabel('Feature Extractor')
ax.set_xlabel('Time (ms)')
ax.set_title('Average time to compute ~300 Key-Point Descriptors')
plt.show()

"""# Compute total number of features"""

# SIFT Timing
sift = cv.xfeatures2d.SIFT_create()
total_time=0
total_features=0
for imgname in files:
    img = cv.imread(os.path.join(img_folder, imgname))
    start_time=time.time()
    kp,des = sift.detectAndCompute(img,None)
    total_time+=time.time()-start_time
    total_features+=np.array(kp).shape[0]
sift_time2=total_time/2
sift_features2 = total_features/2
print("Average time for SIFT features: ",sift_time2)
print("Average number of SIFT features: ",sift_features2)

# ORB Timing
orb = cv.ORB_create(1000000)
total_time=0
total_features=0
for imgname in files:
    img = cv.imread(os.path.join(img_folder, imgname))
    start_time=time.time()
    kp,des = orb.detectAndCompute(img,None)
    total_time+=time.time()-start_time
    total_features+=np.array(kp).shape[0]
orb_time2=total_time/2
orb_features2 = total_features/2

print("Average time for ORB features: ",orb_time2)
print("Average number of ORB features: ",orb_features2)

fig = plt.figure()
sns.set()
ax = fig.add_axes([0,0,1,1])
methods = ['SIFT',  'ORB']
times = [sift_features2,orb_features2]
ax.barh(methods,times,color=('green','orange'))
ax.set_ylabel('Feature Extractor')
ax.set_xlabel('Number of key-points')
ax.set_title('Average total number of extracted key-points per image')
plt.show()

reduce= True

# SIFT Brightness Matching
sift = cv.xfeatures2d.SIFT_create(1000000)
total_time=0
total_features=0
bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
p_matched = []
dist_list = []

for imgname in files:
    img1 = cv.imread(os.path.join(img_folder, "road.png"))
    img2 = cv.imread(os.path.join(img_folder, "road2.png"))
    kp1,des1 = sift.detectAndCompute(img1,None)
    kp2,des2 = sift.detectAndCompute(img2,None)
    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)
    p_matched.append(len(matches)/len(kp1))

    if reduce:
        matches = matches[:500]
    distance = []
    for match in matches:
        c1 = kp1[match.queryIdx].pt
        c2 = kp2[match.trainIdx].pt
        dist = ((c1[0]-c2[0])**2+(c1[1]-c2[1])**2)**0.5
        distance.append(dist)
    dist_list.append(np.average(dist))
sift_per_b = np.average(p_matched)
sift_dist_b = np.average(dist_list)
print('Percentage of Matched Keypoints: ', sift_per_b)
print('Drift of Matched Keypoints: ', sift_dist_b)

# SIFT Rotation Matching
sift = cv.xfeatures2d.SIFT_create(1000000)
total_time=0
total_features=0
bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
p_matched = []
dist_list = []

for imgname in files:

    img1 = cv.imread(os.path.join(img_folder, "road.png"))
    img2 = cv.imread(os.path.join(img_folder, "road.png"))
    img2 = cv.rotate(img2, cv.ROTATE_90_CLOCKWISE)
    kp1,des1 = sift.detectAndCompute(img1,None)
    kp2,des2 = sift.detectAndCompute(img2,None)
    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)
    p_matched.append(len(matches)/len(kp1))

    if reduce:
        matches = matches[:500]
    distance = []
    for match in matches:
        c1 = kp1[match.queryIdx].pt
        c2 = kp2[match.trainIdx].pt
        dist = ((c1[0]-c2[0])**2+(c1[1]-c2[1])**2)**0.5
        distance.append(dist)
    dist_list.append(np.average(dist))
sift_per_r = np.average(p_matched)
sift_dist_r = np.average(dist_list)
print('Percentage of Matched Keypoints: ', sift_per_r)
print('Drift of Matched Keypoints: ', sift_dist_r)


# ORB Brightness Matching
orb = cv.ORB_create(1000000)
total_time=0
total_features=0
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
p_matched = []
dist_list = []

for imgname in files:
    img1 = cv.imread(os.path.join(img_folder, "road.png"))
    img2 = cv.imread(os.path.join(img_folder, "road2.png"))
    kp1,des1 = orb.detectAndCompute(img1,None)
    kp2,des2 = orb.detectAndCompute(img2,None)
    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)
    p_matched.append(len(matches)/len(kp1))

    if reduce:
        matches = matches[:500]    
    distance = []
    for match in matches:
        c1 = kp1[match.queryIdx].pt
        c2 = kp2[match.trainIdx].pt
        dist = ((c1[0]-c2[0])**2+(c1[1]-c2[1])**2)**0.5
        distance.append(dist)
    dist_list.append(np.average(dist))
orb_per_b = np.average(p_matched)
orb_dist_b = np.average(dist_list)
print('Percentage of Matched Keypoints: ', orb_per_b)
print('Drift of Matched Keypoints: ', orb_dist_b)

# ORB Rotation Matching
orb = cv.ORB_create(1000000)
total_time=0
total_features=0
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
p_matched = []
dist_list = []

for imgname in files:
    img1 = cv.imread(os.path.join(img_folder, "road.png"))
    img2 = cv.imread(os.path.join(img_folder, "road.png"))
    img2 = cv.rotate(img2, cv.ROTATE_90_CLOCKWISE)
    kp1,des1 = orb.detectAndCompute(img1,None)
    kp2,des2 = orb.detectAndCompute(img2,None)
    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)
    p_matched.append(len(matches)/len(kp1))

    if reduce:
        matches = matches[:500]
    distance = []
    for match in matches:
        c1 = kp1[match.queryIdx].pt
        c2 = kp2[match.trainIdx].pt
        dist = ((c1[0]-c2[0])**2+(c1[1]-c2[1])**2)**0.5
        distance.append(dist)
    dist_list.append(np.average(dist))
orb_per_r = np.average(p_matched)
orb_dist_r = np.average(dist_list)
print('Percentage of Matched Keypoints: ', orb_per_r)
print('Drift of Matched Keypoints: ', orb_dist_r)

fig = plt.figure()
sns.set()
ax = fig.add_axes([0,0,1,1])
methods = ['SIFT',  'ORB']
times = [sift_per_b,orb_per_b]
ax.barh(methods,times,color=('green','orange'))
ax.set_ylabel('Feature Extractor')
ax.set_xlabel('Percentage')
ax.set_title('Average Percentage of Matched Keypoints for Brightened Image')
plt.show()

fig = plt.figure()
sns.set()
ax = fig.add_axes([0,0,1,1])
methods = ['SIFT', 'ORB']
times = [sift_per_r,orb_per_r]
ax.barh(methods,times,color=('green','orange'))
ax.set_ylabel('Feature Extractor')
ax.set_xlabel('Percentage')
ax.set_title('Average Percentage of Matched Keypoints for Rotated Image')
plt.show()

img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:500],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3),plt.show()
print("done!")