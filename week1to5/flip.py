import cv2
import numpy as np

img = cv2.imread("images/preview.webp")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
#turn it into pure black-and-white 
#anything brighter than 100 becomes pure white (255), anything darker becomes pure black (0)

contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#if just counting solid objects with no holes then RETR_EXTERNAL
#But if your image has objects WITH holes then RETR_CCOMP or RETR_TREE

result = img.copy()#make a duplicate of the original photo,to draw without messing original
small_count = 0
large_count = 0 #two counters starting at zero, like a tally you will add to as you go.
shape_counts = {'Triangle': 0, 'Square': 0, 'Circle': 0, 'Unknown/Merged': 0}
#this is a dictionary,all starting at zero, ready to be incremented.

#  Analyze each object 
for i, cnt in enumerate(contours): #Go through every traced outline one at a time
    area = cv2.contourArea(cnt)
    # skip tiny noise specks Measure ans how much space this shape covers (its area, in pixels)
    if area < 100:
        continue

    perimeter = cv2.arcLength(cnt, True)#measures the total length of the outline itself
    x, y, w, h = cv2.boundingRect(cnt)#to draw our label box.

    # approximate the contour to count its corners
    approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
    num_corners = len(approx)#counts how many corner points were found

    # SIZE classification (tune 5000 based on YOUR image's data)
    #Deciding Small or Large
    if area < 5000:
        size_label = 'Small'
        small_count += 1
    else:
        size_label = 'Large'
        large_count += 1

    # SHAPE classification using corner count (Deciding the Shape)
    if num_corners == 3:
        shape = 'Triangle'
    elif num_corners == 4:
        shape = 'Square'
    elif 6 <= num_corners <= 9:
        shape = 'Circle'
    else:
        shape = 'Unknown/Merged'   # catches touching/overlapping objects
    shape_counts[shape] += 1

    # Draw label
    color = (0, 0, 255) if shape == 'Unknown/Merged' else (0, 255, 0)
    label = f'{size_label}-{shape}'  #Large-Circle like that
    cv2.rectangle(result, (x, y), (x+w, y+h), color, 2)
    cv2.putText(result, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

# Save and report 
cv2.imwrite('classified_result.png', result)

print('===== SUMMARY REPORT =====')
print(f'Total objects detected: {len(contours)}')
print(f'Size  -> Small: {small_count}, Large: {large_count}')
print(f'Shape -> {shape_counts}')