import cv2
import numpy as np

template = cv2.imread('images/box.png', cv2.IMREAD_GRAYSCALE)
scene = cv2.imread('images/box_scene.png', cv2.IMREAD_GRAYSCALE)

orb = cv2.ORB_create(1000)
kp1, des1 = orb.detectAndCompute(template, None)#find those distictive spots and then create 
kp2, des2 = orb.detectAndCompute(scene, None)#numeric fingerprint that describe specific spot(unique id)

#check every single fingerprint from image 1 against every single fingerprint from image 2, one by one, no shortcuts.
bf = cv2.BFMatcher(cv2.NORM_HAMMING)#Brute Force Matcher does the actual comparison job 
raw_matches = bf.knnMatch(des1, des2, k=2)
#For every single fingerprint in des1 (Image 1's fingerprints), 
#don't just find its ONE single best match in des2 find its top 2 best matches instead.
good_matches = []
for m, n in raw_matches:
    if m.distance < 0.75 * n.distance:#is m's distance less than 75% of n's distance
        good_matches.append(m)

print('Good matches:', len(good_matches))#in that two matches append the best one(m or n)

pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
#For every good match, look up where its point is located in image 1, 
#collect all those positions into one list,and package it in the format findHomography needs
pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
#Same exact idea, but this time looking up the corresponding positions in image 2
H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
print(f'RANSAC inliers: {mask.ravel().sum()} out of {len(mask)}')

h, w = template.shape
corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
transformed_corners = cv2.perspectiveTransform(corners, H)

scene_color = cv2.cvtColor(scene, cv2.COLOR_GRAY2BGR)
cv2.polylines(scene_color, [np.int32(transformed_corners)], True, (0,255,0), 3)
cv2.imwrite('output/box_result.png', scene_color)