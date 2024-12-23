# CIFAR-10_Image_Classification_with_CNN

This project demonstrates the implementation of a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset into one of 10 categories. The workflow includes data preprocessing, augmentation, model training, and evaluation. This notebook is ideal for understanding image classification tasks using deep learning techniques.

# Project_Overview
The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images. This project builds a CNN model using TensorFlow/Keras to achieve a high classification accuracy on this dataset.

# Dataset_Information
- **Dataset**: CIFAR-10
- **Classes**: Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck
- **Image Size**: 32x32 pixels

# Project_Structure
1. **Loading CIFAR-10 Dataset**: The dataset is loaded and split into training and testing sets.
2. **Preprocessing the Data**: Images are normalized to improve model performance.
3. **Data Augmentation**: Techniques like random flips and rotations are applied to enrich the dataset and reduce overfitting.
4. **Building the CNN Model**: A CNN architecture is designed with convolutional, pooling, and fully connected layers.
5. **Compiling the Model**: The model is compiled with an optimizer, loss function, and evaluation metric.
6. **Training the Model**: The model is trained on the dataset with a learning rate scheduler and early stopping.
7. **Evaluating the Model**: Test accuracy and loss are reported to measure performance.

# Results
- **Test Accuracy**: Achieved approximately 90% accuracy on the test set.
- **Test Loss**: Demonstrated effective generalization with a reduced loss value.

# Prerequisites
- Python 3.8 or later
- TensorFlow 2.x
- Jupyter Notebook or any Python IDE

# Instructions_to_Run
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/CIFAR-10-CNN-Classifier.git
   ```
2. Navigate to the project directory:
   ```bash
   cd CIFAR-10-CNN-Classifier
   ```
3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Open the notebook:
   ```bash
   jupyter notebook "Image Classification with CNN.ipynb"
   ```
5. Run all cells in the notebook to reproduce results.

# Model_Summary
The CNN model includes the following layers:
- **Convolutional Layers**: Extract features from images
- **Pooling Layers**: Reduce dimensionality while retaining essential features
- **Fully Connected Layers**: Perform classification based on extracted features
- **Dropout Layers**: Prevent overfitting during training

# Conclusion
This project successfully classifies CIFAR-10 images with high accuracy, demonstrating the effectiveness of CNNs in image recognition tasks. The use of data augmentation and learning rate scheduling significantly contributed to the model's performance.

# Future_Work
- Implement a web-based frontend for live predictions.
- Experiment with transfer learning using pretrained models.
- Extend to CIFAR-100 for more complex classification tasks.

# License
This project is licensed under the MIT License. See the LICENSE file for details.

---

Feel free to contribute to this project or use it as a learning resource!