import cv2
import numpy as np

# Read input images
photoA = cv2.imread("images/photoA.png")
photoB = cv2.imread("images/photoB.png")

# Check if images were loaded correctly
if photoA is None:
    raise FileNotFoundError("Could not load images/photoA.png")

if photoB is None:
    raise FileNotFoundError("Could not load images/photoB.png")

# Convert images to grayscale
grayA = cv2.cvtColor(photoA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(photoB, cv2.COLOR_BGR2GRAY)

# Create ORB detector
orb = cv2.ORB_create(500)

# Detect keypoints and compute descriptors
kpA, desA = orb.detectAndCompute(grayA, None)
kpB, desB = orb.detectAndCompute(grayB, None)

print("Landmarks found in photo A:", len(kpA))
print("Landmarks found in photo B:", len(kpB))

# Brute Force Matcher using Hamming distance
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors
matches = bf.match(desA, desB)

# Sort matches by distance
matches = sorted(matches, key=lambda x: x.distance)

# Select the 40 best matches
good_matches = matches[:40]

print("Using", len(good_matches), "best matches for alignment")

# Draw the first 20 matches
match_img = cv2.drawMatches(
    photoA,
    kpA,
    photoB,
    kpB,
    good_matches[:20],
    None,
    flags=2
)

# Save feature matching result
cv2.imwrite("output/orb_matches.png", match_img)

# Get matching points
ptsA = np.float32([
    kpA[m.queryIdx].pt
    for m in good_matches
])

ptsB = np.float32([
    kpB[m.trainIdx].pt
    for m in good_matches
])

# Calculate homography using RANSAC
H, mask = cv2.findHomography(
    ptsB,
    ptsA,
    cv2.RANSAC,
    5.0
)

if H is None:
    raise RuntimeError("Homography could not be calculated.")

print(
    f"RANSAC inliers: {mask.ravel().sum()} "
    f"out of {len(mask)}"
)

# Create panorama canvas
result_width = photoA.shape[1] + photoB.shape[1]
result_height = max(photoA.shape[0], photoB.shape[0])

# Warp photo B into photo A's coordinate system
result = cv2.warpPerspective(
    photoB,
    H,
    (result_width, result_height),
    borderValue=(240, 240, 240)
)

# Place photo A on the panorama
result[
    0:photoA.shape[0],
    0:photoA.shape[1]
] = photoA

# Save final panorama
cv2.imwrite(
    "output/stitched_resultt.png",
    result
)

print("Panorama saved as output/stitched_resultt.png")