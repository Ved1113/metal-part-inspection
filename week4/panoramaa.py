import cv2
import numpy as np

photoA = cv2.imread("images/photoA.png")
photoB = cv2.imread("images/photoB.png")

grayA = cv2.cvtColor(photoA,cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(photoB,cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create(500)
kpA,desA =orb.detectAndCompute(grayA,None)
kpB,desB = orb.detectAndCompute(grayB,None)

print("landmark found in photo A:",len(kpA))
print("landmark found in photo B:",len(kpB))

bf =cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
matches = bf.match(desA,desB)
matches = sorted(matches,key=lambda x:x.distance)

good_matches = matches[:40]
print('using',len(good_matches),'best matches for allignment')

match_img = cv2.drawMatches(photoA,kpA,photoB,kpB,good_matches[:20],None,flags=2)
cv2.imwrite('orb_matches.png',match_img)


ptsA = np.float32([kpA[m.queryIdx].pt for m in good_matches])
ptsB = np.float32([kpB[m.queryIdx].pt for m in good_matches])

H,mask = cv2.findHomography(ptsB,ptsA,cv2.RANSAC,5.0)
print(f'RANSAC inliers: {mask.ravel().sum()} out of {len(mask)}')
result_width = photoA.shape[1]+photoB.shape[1]
result_height = photoA.shape[0]

result = cv2.warpPerspective(photoB,H,(result_width,result_height),borderValue=(240,240,240))

result[0:photoA.shape[0],0:photoA.shape[1]]= photoA

cv2.imwrite('stitched_resultt.png', result)
print('Panorama saved as stitched_result.png')