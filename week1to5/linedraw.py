import cv2
img = cv2.imread("images/sample.jpg")

pt1=(50,100)
pt2=(300,200) 
colour=(255,0,0)
thickness=4

cv2.line(img,pt1,pt2,colour,thickness)
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()