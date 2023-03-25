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
for imgname in os.listdir(img_folder+'normal/'):
    img = cv.imread(img_folder+'normal/'+imgname)
    start_time=time.time()
    kp,des = sift.detectAndCompute(img,None)
    total_time+=time.time()-start_time
    total_features+=np.array(kp).shape[0]
sift_time=total_time/5
sift_features = total_features/5
print("Average time for SIFT features: ",sift_time)
print("Average number of SIFT features: ",sift_features)

# SURF Timing
surf = cv.xfeatures2d.SURF_create(2200)
total_time=0
total_features=0
for imgname in os.listdir(img_folder+'normal/'):
    img = cv.imread(img_folder+'normal/'+imgname)
    start_time=time.time()
    kp,des = surf.detectAndCompute(img,None)
    total_time+=time.time()-start_time
    total_features+=np.array(kp).shape[0]
surf_time=total_time/5
surf_features = total_features/5

print("Average time for SURF features: ",surf_time)
print("Average number of SURF features: ",surf_features)

# ORB Timing
orb = cv.ORB_create(300)
total_time=0
total_features=0
for imgname in os.listdir(img_folder+'normal/'):
    img = cv.imread(img_folder+'normal/'+imgname)
    start_time=time.time()
    kp,des = orb.detectAndCompute(img,None)
    total_time+=time.time()-start_time
    total_features+=np.array(kp).shape[0]
orb_time=total_time/5
orb_features = total_features/5

print("Average time for ORB features: ",orb_time)
print("Average number of ORB features: ",orb_features)

img2=cv.drawKeypoints(img,kp,img)
plt.imshow(img2)
plt.axis('off')

fig = plt.figure()
sns.set()
ax = fig.add_axes([0,0,1,1])
methods = ['SIFT', 'SURF', 'ORB']
times = [sift_time*1000,surf_time*1000,orb_time*1000]
ax.barh(methods,times,color=('green','blue','orange'))
ax.set_ylabel('Feature Extractor')
ax.set_xlabel('Time (ms)')
ax.set_title('Average time to compute ~300 Key-Point Descriptors')
plt.show()

"""# Compute total number of features"""

# SIFT Timing
sift = cv.xfeatures2d.SIFT_create()
total_time=0
total_features=0
for imgname in os.listdir(img_folder+'normal/'):
    img = cv.imread(img_folder+'normal/'+imgname)
    start_time=time.time()
    kp,des = sift.detectAndCompute(img,None)
    total_time+=time.time()-start_time
    total_features+=np.array(kp).shape[0]
sift_time2=total_time/5
sift_features2 = total_features/5
print("Average time for SIFT features: ",sift_time2)
print("Average number of SIFT features: ",sift_features2)

# SURF Timing
surf = cv.xfeatures2d.SURF_create()
total_time=0
total_features=0
for imgname in os.listdir(img_folder+'normal/'):
    img = cv.imread(img_folder+'normal/'+imgname)
    start_time=time.time()
    kp,des = surf.detectAndCompute(img,None)
    total_time+=time.time()-start_time
    total_features+=np.array(kp).shape[0]
surf_time2=total_time/5
surf_features2 = total_features/5

print("Average time for SURF features: ",surf_time2)
print("Average number of SURF features: ",surf_features2)

# ORB Timing
orb = cv.ORB_create(1000000)
total_time=0
total_features=0
for imgname in os.listdir(img_folder+'normal/'):
    img = cv.imread(img_folder+'normal/'+imgname)
    start_time=time.time()
    kp,des = orb.detectAndCompute(img,None)
    total_time+=time.time()-start_time
    total_features+=np.array(kp).shape[0]
orb_time2=total_time/5
orb_features2 = total_features/5

print("Average time for ORB features: ",orb_time2)
print("Average number of ORB features: ",orb_features2)

fig = plt.figure()
sns.set()
ax = fig.add_axes([0,0,1,1])
methods = ['SIFT', 'SURF', 'ORB']
times = [sift_features2,surf_features2,orb_features2]
ax.barh(methods,times,color=('green','blue','orange'))
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

for imgname in os.listdir(img_folder+'normal/'):

    img1 = cv.imread(img_folder+'normal/'+imgname)
    img2 = cv.imread(img_folder+'brightness/'+imgname)
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

for imgname in os.listdir(img_folder+'normal/'):

    img1 = cv.imread(img_folder+'normal/'+imgname)
    img2 = cv.imread(img_folder+'rotate/'+imgname)
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
        dist = ((c1[0]-(c2[0]*(-1)+720))**2+(c1[1]-(c2[1]*(-1)+480))**2)**0.5
        distance.append(dist)
    dist_list.append(np.average(dist))
sift_per_r = np.average(p_matched)
sift_dist_r = np.average(dist_list)
print('Percentage of Matched Keypoints: ', sift_per_r)
print('Drift of Matched Keypoints: ', sift_dist_r)

# SURF Brightness Matching
surf = cv.xfeatures2d.SURF_create()
total_time=0
total_features=0
bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
p_matched = []
dist_list = []

for imgname in os.listdir(img_folder+'normal/'):

    img1 = cv.imread(img_folder+'normal/'+imgname)
    img2 = cv.imread(img_folder+'brightness/'+imgname)
    kp1,des1 = surf.detectAndCompute(img1,None)
    kp2,des2 = surf.detectAndCompute(img2,None)
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
surf_per_b = np.average(p_matched)
surf_dist_b = np.average(dist_list)
print('Percentage of Matched Keypoints: ', surf_per_b)
print('Drift of Matched Keypoints: ', surf_dist_b)

# SURF Rotation Matching
surf = cv.xfeatures2d.SURF_create()
total_time=0
total_features=0
bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
p_matched = []
dist_list = []

for imgname in os.listdir(img_folder+'normal/'):

    img1 = cv.imread(img_folder+'normal/'+imgname)
    img2 = cv.imread(img_folder+'rotate/'+imgname)
    kp1,des1 = surf.detectAndCompute(img1,None)
    kp2,des2 = surf.detectAndCompute(img2,None)
    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)
    p_matched.append(len(matches)/len(kp1))

    if reduce:
        matches = matches[:500]
    distance = []
    for match in matches:
        c1 = kp1[match.queryIdx].pt
        c2 = kp2[match.trainIdx].pt
        dist = ((c1[0]-(c2[0]*(-1)+720))**2+(c1[1]-(c2[1]*(-1)+480))**2)**0.5
        distance.append(dist)
    dist_list.append(np.average(dist))
surf_per_r = np.average(p_matched)
surf_dist_r = np.average(dist_list)
print('Percentage of Matched Keypoints: ', surf_per_r)
print('Drift of Matched Keypoints: ', surf_dist_r)

# ORB Brightness Matching
orb = cv.ORB_create(1000000)
total_time=0
total_features=0
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
p_matched = []
dist_list = []

for imgname in os.listdir(img_folder+'normal/'):

    img1 = cv.imread(img_folder+'normal/'+imgname)
    img2 = cv.imread(img_folder+'brightness/'+imgname)
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

for imgname in os.listdir(img_folder+'normal/'):

    img1 = cv.imread(img_folder+'normal/'+imgname)
    img2 = cv.imread(img_folder+'rotate/'+imgname)
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
        dist = ((c1[0]-(c2[0]*(-1)+720))**2+(c1[1]-(c2[1]*(-1)+480))**2)**0.5
        distance.append(dist)
    dist_list.append(np.average(dist))
orb_per_r = np.average(p_matched)
orb_dist_r = np.average(dist_list)
print('Percentage of Matched Keypoints: ', orb_per_r)
print('Drift of Matched Keypoints: ', orb_dist_r)

fig = plt.figure()
sns.set()
ax = fig.add_axes([0,0,1,1])
methods = ['SIFT', 'SURF', 'ORB']
times = [sift_per_b,surf_per_b,orb_per_b]
ax.barh(methods,times,color=('green','blue','orange'))
ax.set_ylabel('Feature Extractor')
ax.set_xlabel('Percentage')
ax.set_title('Average Percentage of Matched Keypoints for Brightened Image')
plt.show()

fig = plt.figure()
sns.set()
ax = fig.add_axes([0,0,1,1])
methods = ['SIFT', 'SURF', 'ORB']
times = [sift_per_r,surf_per_r,orb_per_r]
ax.barh(methods,times,color=('green','blue','orange'))
ax.set_ylabel('Feature Extractor')
ax.set_xlabel('Percentage')
ax.set_title('Average Percentage of Matched Keypoints for Rotated Image')
plt.show()

fig = plt.figure()
sns.set()
ax = fig.add_axes([0,0,1,1])
methods = ['SIFT', 'SURF', 'ORB']
times = [sift_dist_b,surf_dist_b,orb_dist_b]
ax.barh(methods,times,color=('green','blue','orange'))
ax.set_ylabel('Feature Extractor')
ax.set_xlabel('Pixels')
ax.set_title('Average Drift of Matched Keypoints for Brightened Image')
plt.show()

fig = plt.figure()
sns.set()
ax = fig.add_axes([0,0,1,1])
methods = ['SIFT', 'SURF', 'ORB']
times = [sift_dist_r,surf_dist_r,orb_dist_r]
ax.barh(methods,times,color=('green','blue','orange'))
ax.set_ylabel('Feature Extractor')
ax.set_xlabel('Pixels')
ax.set_title('Average Drift of Matched Keypoints for Rotated Image')
plt.show()

img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:500],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3),plt.show()

# SURF Brightness Matching

surf = cv.xfeatures2d.SURF_create()
bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)

img1 = cv.imread(img_folder+'normal/'+imgname)
img2 = cv.imread(img_folder+'brightness/'+imgname)
kp1,des1 = surf.detectAndCompute(img1,None)
kp2,des2 = surf.detectAndCompute(img2,None)
matches = bf.match(des1,des2)
matches = sorted(matches, key = lambda x:x.distance)

img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:500],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3),plt.axis('off'),plt.show()

img=cv.drawKeypoints(img1,kp1[:500],img1,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.imshow(img),plt.axis('off'),plt.show()

cv.imwrite(img_folder+'normal/'+'sift_keypoints.jpg',img)
cv.imwrite(img_folder+'normal/'+'sift_matching.jpg',img3)

