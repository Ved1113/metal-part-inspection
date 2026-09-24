# import os

# dataset_path = r"C:\Users\OM\Downloads\archive (4)\NEU-DET\train\images"

# classes = os.listdir(dataset_path)

# print("Classes:")
# for cls in classes:
#     print(cls)
# import os

# dataset_path = r"C:\Users\OM\Downloads\archive (4)\NEU-DET\train\images"

# classes = os.listdir(dataset_path)

# for cls in classes:
#     class_path = os.path.join(dataset_path, cls)
#     images = os.listdir(class_path)

#     print(cls, ":", len(images), "images")
# from PIL import Image
# import os

# dataset_path = r"C:\Users\OM\Downloads\archive (4)\NEU-DET\train\images"

# # Choose one class
# class_name = "patches"

# # Go to that class folder
# class_path = os.path.join(dataset_path, class_name)

# # Get image files
# images = os.listdir(class_path)

# # Take first image
# image_name = images[0]

# # Full image path
# image_path = os.path.join(class_path, image_name)

# # Load image
# image = Image.open(image_path)

# print("Image name:", image_name)
# print("Image size:", image.size)
# print("Image mode:", image.mode)

# # Display image
# image.show()
# import numpy as np

# image_array = np.array(image)

# print("Array shape:", image_array.shape)
# print("Data type:", image_array.dtype)

# print("First pixel:", image_array[0, 0])
# print("First 5 pixels:")
# print(image_array[0, :5])

# import torch
# from torchvision import transforms

# transform = transforms.ToTensor()

# image_tensor = transform(image)

# print("Tensor shape:", image_tensor.shape)
# print("Tensor data type:", image_tensor.dtype)
# print("Minimum value:", image_tensor.min())
# print("Maximum value:", image_tensor.max())
from torchvision import datasets, transforms

transform = transforms.ToTensor()

dataset = datasets.ImageFolder(
    root=r"C:\Users\OM\Downloads\archive (4)\NEU-DET\train\images",
    transform=transform
)

# print("Classes:", dataset.classes)
# print("Number of images:", len(dataset))
# image, label = dataset[0]

# print("Image shape:", image.shape)
# print("Label:", label)
# print("Class:", dataset.classes[label])
# from torch.utils.data import DataLoader

# train_loader = DataLoader(
#     dataset,
#     batch_size=32,
#     shuffle=True
# )

# images, labels = next(iter(train_loader))

# print("Images shape:", images.shape)
# print("Labels shape:", labels.shape)
# print("Labels:", labels)
# import matplotlib.pyplot as plt

# images, labels = next(iter(train_loader))

# plt.figure(figsize=(10, 10))

# for i in range(16):
#     plt.subplot(4, 4, i + 1)

#     # CHW → HWC for matplotlib
#     img = images[i].permute(1, 2, 0)

#     plt.imshow(img)
#     plt.title(dataset.classes[labels[i]])
#     plt.axis("off")

# plt.tight_layout()
# plt.show()
# import torch
# import torch.nn as nn
# class SimpleCNN(nn.Module):

#     def __init__(self):
#         super().__init__()

#         self.features = nn.Sequential(

#             # First convolution
#             nn.Conv2d(
#                 in_channels=3,
#                 out_channels=16,
#                 kernel_size=3,
#                 padding=1
#             ),

#             nn.ReLU(),

#             nn.MaxPool2d(kernel_size=2),

#             # Second convolution
#             nn.Conv2d(
#                 in_channels=16,
#                 out_channels=32,
#                 kernel_size=3,
#                 padding=1
#             ),

#             nn.ReLU(),

#             nn.MaxPool2d(kernel_size=2)
#         )

#         self.classifier = nn.Sequential(

#             nn.Flatten(),

#             nn.Linear(32 * 50 * 50, 128),

#             nn.ReLU(),

#             nn.Linear(128, 6)
#         )

#     def forward(self, x):

#         x = self.features(x)

#         x = self.classifier(x)

#         return x
model = SimpleCNN()

print(model)
    