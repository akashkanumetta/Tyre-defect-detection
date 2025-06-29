## Tyre Defect Detection Using Convolutional Neural Networks

This project focuses on classifying tyre images as either defective or in good condition using a convolutional neural network (CNN). The model is trained on a labeled dataset containing images of tyres with visible defects and tyres in normal condition. It serves as a foundation for intelligent visual inspection systems in automotive applications.

Project Overview

A custom CNN architecture is designed and trained using Keras with TensorFlow backend. The model takes in tyre images, processes them through several convolutional and pooling layers, and classifies them into two categories: defective or good. Data augmentation is used to improve generalization and robustness against variations in angle, lighting, and noise.

Features

* End-to-end defect classification using deep learning
* Data augmentation to simulate real-world scenarios
* Custom CNN with dropout and batch normalization for better performance
* Model training history plotted and evaluated
* Trained model saved in HDF5 format for reuse

Dataset

The dataset contains two folders: one for defective tyres and one for good-condition tyres. Images are captured in varying resolutions and lighting conditions to reflect real-world inspection environments. All images are resized to 128x128 pixels and normalized for input to the CNN.

Each image is labeled as:

* 1 for defective
* 0 for good

Data is split into training and validation sets with an 80:20 ratio.

Technologies Used

* Python
* TensorFlow and Keras for model development
* NumPy and Matplotlib for data handling and visualization
* ImageDataGenerator for augmentation and preprocessing

Use Cases

* Automated tyre quality inspection in manufacturing units
* Roadside vehicle safety checks using image capture
* Integration with mobile or embedded systems for real-time analysis

Future Improvements

* Use of pretrained models like MobileNet or ResNet for improved accuracy
* Extension to multi-class classification (e.g., tread wear, cracks, bulges)
* Integration with FastAPI or Flask for API-based inspection services
* Deployment on edge devices like Raspberry Pi or Jetson Nano
