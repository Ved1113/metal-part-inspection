import cv2
import numpy as np

# ============ STEP 1: Load scene and template ============
img = cv2.imread('images/parts_scene.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
template = cv2.imread('images/nut_template.png', cv2.IMREAD_GRAYSCALE)
h, w = template.shape

# ============ STEP 2: Search the whole image for matches ============
result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)

threshold = 0.65   # tuned based on testing - separates real matches from noise
locations = np.where(result >= threshold)

boxes, scores = [], []
for pt in zip(*locations[::-1]):
    boxes.append([pt[0], pt[1], pt[0] + w, pt[1] + h])
    scores.append(float(result[pt[1], pt[0]]))

print('Raw candidate detections:', len(boxes))

# ============ STEP 3: Remove duplicate/overlapping detections (NMS) ============
indices = cv2.dnn.NMSBoxes(
    bboxes=[[int(b[0]), int(b[1]), int(b[2]-b[0]), int(b[3]-b[1])] for b in boxes],
    scores=scores, score_threshold=threshold, nms_threshold=0.2
)
print('Final detections after NMS:', len(indices))

# ============ STEP 4: Draw and save results ============
result_img = img.copy()
for i in indices:
    i = i if isinstance(i, (int, np.integer)) else i[0]
    x1, y1, x2, y2 = boxes[i]
    conf = scores[i]
    cv2.rectangle(result_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
    cv2.putText(result_img, f'{conf:.2f}', (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

cv2.imwrite('parts_result.png', result_img)
print('Saved as parts_result.png')

# NOTE: I also tried combining this with a contour-based "hole check" to
# distinguish nuts from bolt heads (which look similar in shape). This did
# NOT work on this image, because the white background blends with the
# bright metallic parts at any threshold value - there's not enough contrast
# to separate object from background using simple thresholding here. This
# only works well when there's strong contrast between object and background
# (like our earlier washer example on a dark background).