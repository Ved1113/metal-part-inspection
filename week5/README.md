# Week 5 — Deep Learning Foundations & Dataset Building

## Objective

This week focuses on the fundamentals of deep learning for industrial metal-part inspection.

The work covers CNN basics, industrial image datasets, data augmentation, class imbalance, and transfer learning using pretrained CNN models.

The main classification task is:

**Good vs Defective**

---

## 1. CNN Fundamentals

A basic CNN was implemented to understand how convolutional neural networks process images.

Topics covered:

- Image tensors and batches
- Convolution layers
- ReLU activation
- Max pooling
- Fully connected layers
- Forward propagation
- Loss calculation
- Backpropagation
- Optimizer and weight updates
- Training over multiple epochs

A SimpleCNN model was trained and evaluated on the NEU-DET dataset to understand the complete CNN training workflow.

**Notebook:** `cnndata.ipynb`

---

##  Industrial Metal Defect Dataset

The NEU-DET metal surface defect dataset was explored and used for CNN experiments.

The dataset contains six defect classes:

- Crazing
- Inclusion
- Patches
- Pitted Surface
- Rolled-in Scale
- Scratches

The dataset was inspected by:

- Checking class names
- Counting images per class
- Loading sample images
- Checking image dimensions
- Converting images into NumPy arrays
- Creating PyTorch datasets and dataloaders
- Visualizing image batches

---

## Good vs Defective Dataset

A separate metal-nut dataset was prepared for the binary industrial inspection task.

Classes:

- Good
- Defective

Defective samples include different defect types such as:

- Bent
- Color
- Flip
- Scratch

The different defect types are grouped under the **Defective** class because the final inspection decision is whether the part is acceptable or defective.

**Notebook:** `metal_defect.ipynb` / `mobilev2.ipynb`

---

## Dataset Split

The metal-nut dataset was divided into:

- Training set — 80%
- Validation set — 10%
- Test set — 10%

Stratified splitting was used to maintain the class distribution across the splits.

---

## Data Augmentation

Training images were augmented to improve model generalization and simulate variations that can occur during industrial inspection.

Augmentations include:

- Random horizontal flipping
- Random rotation
- Brightness variation
- Contrast variation
- Gaussian blur

Images were resized to `224 × 224` and normalized using ImageNet normalization values for pretrained models.

---

##  Class Imbalance

The Good and Defective classes are not perfectly balanced.

Class distribution was inspected and class weights were used during training so that the model gives appropriate importance to the minority class.

Weighted cross-entropy loss was used for the MobileNetV2 experiment.

This is important in industrial inspection because missing defective parts can be more costly than incorrectly flagging a good part.

---

## Transfer Learning — ResNet18

A pretrained ResNet18 model was used as a transfer-learning backbone.

The pretrained feature layers were initially frozen and the final classification layer was replaced for the target classification task.

The model was trained using the metal-part dataset and evaluated using:

- Accuracy
- Precision
- Recall
- Confusion matrix
- Classification report

**Notebook:** `cnndata.ipynb` / `metal_defect.ipynb`

---

##  Transfer Learning — MobileNetV2

A pretrained MobileNetV2 model was used for the Good vs Defective classification task.

The pretrained feature extractor was frozen and the final classifier was replaced with a two-class output:

```text
0 → Good
1 → Defective



# Evaluation

The trained models were evaluated using:

Accuracy

Measures the overall percentage of correctly classified images.

Precision

Measures how many images predicted as defective were actually defective.

Recall

Measures how many actual defective images were successfully detected.

Recall is particularly important for industrial inspection because defective parts should not be missed.

Confusion Matrix

Shows the number of correct and incorrect predictions for each class.

Classification Report

Provides precision, recall and F1-score for each class.

# Files
week5/
│
├── cnndata.ipynb
├── metal_defect.ipynb
├── mobilev2.ipynb
└── README.md