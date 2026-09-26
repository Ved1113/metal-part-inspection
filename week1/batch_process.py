import cv2
import os

input_folder = "images/batch_input"
output_folder = "output/batch"

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):

    input_path = os.path.join(input_folder, filename)

    image = cv2.imread(input_path)

    if image is None:
        print(f"Skipping: {filename}")
        continue

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize
    resized = cv2.resize(gray, (500, 500))

    # Save output
    output_path = os.path.join(output_folder, filename)

    cv2.imwrite(output_path, resized)

    print(f"Processed: {filename}")

print("Batch processing completed.")