import cv2
import numpy as np

img = cv2.imread('images/shape.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blurred = cv2.medianBlur(gray, 5)

_, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

cv2.imwrite('temp_otsu.png', otsu)
otsu = cv2.imread('temp_otsu.png', cv2.IMREAD_GRAYSCALE)

contours, _ = cv2.findContours(otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

real_parts = []
for c in contours:
    area = cv2.contourArea(c)
    x, y, w, h = cv2.boundingRect(c)
    if area > 300 and x > 20 and 30 < y < 350:
        real_parts.append(c)

print('Spring pieces found:', len(real_parts))

all_points = np.vstack(real_parts)
x, y, w, h = cv2.boundingRect(all_points)
total_area = sum(cv2.contourArea(c) for c in real_parts)

result = img.copy()
cv2.drawContours(result, real_parts, -1, (0, 255, 0), 3)
cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 3)
cv2.putText(result, f'W={w} H={h} Area={int(total_area)}', (x, y - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

cv2.imwrite('spring_result.png', result)

perimeter = cv2.arcLength(all_points, True)
circularity = (4 * np.pi * total_area) / (perimeter ** 2) if perimeter > 0 else 0

if circularity > 0.85:
    shape_type = 'Circle'
elif circularity > 0.65:
    shape_type = 'Square/Rectangle'
else:
    shape_type = 'Irregular/Complex (not a basic geometric shape)'

print('===== FINAL REPORT =====')
print(f'Objects detected: 1')
print(f'Bounding box: width={w}, height={h}')
print(f'Total area: {total_area:.0f}')
print(f'Circularity score: {circularity:.2f}')
print(f'Shape classification: {shape_type}')