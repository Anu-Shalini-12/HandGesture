# HandGesture
# Hand Gesture Recognition using CNN

This project aims to recognize hand gestures using a Convolutional Neural Network (CNN). The CNN is trained on a dataset of hand gesture images to classify gestures into different categories.

## Libraries Used

- [Keras](https://keras.io/)
- [TensorFlow](https://www.tensorflow.org/)
- [Pillow](https://pillow.readthedocs.io/en/stable/)

```bash
# Install required libraries
pip install keras tensorflow Pillow

## **Dataset Definition**

The dataset used in this project is organized into two main directories:

1. **Training Dataset (`HandGestureDataset/train`):**
   - This directory contains training images for the hand gesture recognition model.
   - Images are categorized into subdirectories based on the corresponding gesture class: **'NONE'**, **'ONE'**, **'TWO'**, **'THREE'**, **'FOUR'**, and **'FIVE'**.
   - The images in each class directory are used for training the Convolutional Neural Network (CNN).

2. **Testing Dataset (`HandGestureDataset/test`):**
   - This directory contains separate testing images for evaluating the trained model.
   - Similar to the training dataset, images are organized into subdirectories for each gesture class: **'NONE'**, **'ONE'**, **'TWO'**, **'THREE'**, **'FOUR'**, and **'FIVE'**.
   - The testing dataset is essential for assessing the model's performance on unseen data.

### Image Characteristics:

- **Color Mode:**
  - Grayscale: Images are represented in grayscale to simplify the input channels for the CNN.

- **Image Size:**
  - Resized to (256, 256): All images are resized to a consistent size to ensure uniformity in input dimensions for the neural network.

### Example Directory Structure:

HandGestureDataset/
|-- train/
| |-- NONE/
| | |-- gesture1.png
| | |-- gesture2.png
| | |-- ...
| |
| |-- ONE/
| | |-- gesture1.png
| | |-- gesture2.png
| | |-- ...
| |
| |-- TWO/
| | |-- gesture1.png
| | |-- gesture2.png
| | |-- ...
| |
| |-- THREE/
| | |-- gesture1.png
| | |-- gesture2.png
| | |-- ...
| |
| |-- FOUR/
| | |-- gesture1.png
| | |-- gesture2.png
| | |-- ...
| |
| |-- FIVE/
| |-- gesture1.png
| |-- gesture2.png
| |-- ...
|
|-- test/
|-- NONE/
| |-- gesture1.png
| |-- gesture2.png
| |-- ...
|
|-- ONE/
| |-- gesture1.png
| |-- gesture2.png
| |-- ...
|
|-- TWO/
| |-- gesture1.png
| |-- gesture2.png
| |-- ...
|
|-- THREE/
| |-- gesture1.png
| |-- gesture2.png
| |-- ...
|
|-- FOUR/
| |-- gesture1.png
| |-- gesture2.png
| |-- ...
|
|-- FIVE/
|-- gesture1.png
|-- gesture2.png
|-- ...


This structure helps maintain an organized and labeled dataset, facilitating efficient training and evaluation of the hand gesture recognition model.

## **Training and Evaluation**

The model is trained using the `train.py` script, leveraging a Convolutional Neural Network (CNN) architecture implemented in Keras. The training process includes data augmentation using ImageDataGenerator to enhance model generalization. After training, the model's performance is assessed on the test set.

```bash
# Run the training script
python train.py

##**Save the Model**:
The best-performing model is automatically saved during training with the following details:
-Model Architecture: 'model.json'
-Model Weights: 'model.h5'

## **Classifying Hand Gesture Images**

To classify hand gesture images using the provided test script, follow these steps:

1. **Ensure the `check` Directory:**
   - Make sure that the `check` directory contains the hand gesture images you want to classify.

2. **Run the Test Script:**
   - Open a terminal or command prompt.

   ```bash
   python test.py

Make sure to replace the correct path to your 'check' directory.

##Author:
Anu-Shalini-12




