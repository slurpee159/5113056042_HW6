
# README for HW6: Transfer Learning with VGG16 and Hugging Face Integration

## 1. Prompt: How can I implement transfer learning using VGG16?

### Question:
What are the steps to use VGG16 for transfer learning in classifying images of medical masks?

### Solution:
The VGG16 model is imported from TensorFlow, and its pre-trained weights are frozen to focus training on custom layers added for binary classification.

#### Code Snippet:
```python
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models

vgg16_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
vgg16_base.trainable = False

model = models.Sequential([
    vgg16_base,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(2, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

---

## 2. Prompt: How should I prepare my dataset for training?

### Question:
What steps should I take to preprocess and organize my dataset?

### Solution:
The dataset is dynamically split into training and testing directories with random allocation of images into `train` and `test` subfolders.

#### Code Snippet:
```python
import os
import shutil
import random

def split_dataset(base_path, train_ratio=0.8):
    train_path = os.path.join(base_path, 'train')
    test_path = os.path.join(base_path, 'test')
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    for category in ['with_mask', 'without_mask']:
        category_path = os.path.join(base_path, category)
        images = os.listdir(category_path)
        random.shuffle(images)

        split_idx = int(len(images) * train_ratio)
        train_images = images[:split_idx]
        test_images = images[split_idx:]

        for img in train_images:
            shutil.copy(os.path.join(category_path, img), os.path.join(train_path, category, img))
        for img in test_images:
            shutil.copy(os.path.join(category_path, img), os.path.join(test_path, category, img))

    return train_path, test_path
```

---

## 3. Prompt: How can I evaluate the model's performance?

### Question:
What steps are required to train and evaluate the VGG16 model?

### Solution:
We train the model with an augmented dataset and validate its performance on unseen test data.

#### Code Snippet:
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_path, target_size=(224, 224), batch_size=32, class_mode='categorical'
)
test_generator = test_datagen.flow_from_directory(
    test_path, target_size=(224, 224), batch_size=32, class_mode='categorical'
)

model.fit(train_generator, epochs=3, validation_data=test_generator)
```

---

## 4. Prompt: How can I classify an image with the trained model?

### Question:
How do I use the trained VGG16 model to classify an image into `with_mask` or `without_mask`?

### Solution:
Preprocess the image and predict its class using the trained model.

#### Code Snippet:
```python
import numpy as np
from PIL import Image

def classify_image(image_path, model, class_labels):
    img = Image.open(image_path).resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]
    print(f"Predicted Class: {predicted_class}")
```

---

## 5. Prompt: How can I use Hugging Face for text classification?

### Question:
Can Hugging Face's `pipeline` be used to analyze text sentiment?

### Solution:
Using the `transformers` library, a pre-trained model can classify text input into categories such as positive or negative sentiment.

#### Code Snippet:
```python
from transformers import pipeline

def text_classification():
    classifier = pipeline("text-classification")
    text = input("Enter a sentence for classification: ")
    result = classifier(text)
    print(f"Classification Result: {result}")
```

---

## 6. Final Program

### Complete Code:
```python
# Import necessary libraries
import os
import shutil
import random
import numpy as np
from PIL import Image
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from transformers import pipeline

# Build and compile the model
vgg16_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
vgg16_base.trainable = False

model = models.Sequential([
    vgg16_base,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(2, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Dataset preparation function
def split_dataset(base_path, train_ratio=0.8):
    train_path = os.path.join(base_path, 'train')
    test_path = os.path.join(base_path, 'test')
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    for category in ['with_mask', 'without_mask']:
        category_path = os.path.join(base_path, category)
        images = os.listdir(category_path)
        random.shuffle(images)
        split_idx = int(len(images) * train_ratio)
        train_images, test_images = images[:split_idx], images[split_idx:]
        for img in train_images:
            shutil.copy(os.path.join(category_path, img), os.path.join(train_path, category, img))
        for img in test_images:
            shutil.copy(os.path.join(category_path, img), os.path.join(test_path, category, img))
    return train_path, test_path

# Dataset paths
base_path = './Face-Mask-Detection/dataset'
train_path, test_path = split_dataset(base_path)

# Train the model
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(train_path, target_size=(224, 224), batch_size=32, class_mode='categorical')
test_generator = test_datagen.flow_from_directory(test_path, target_size=(224, 224), batch_size=32, class_mode='categorical')
model.fit(train_generator, epochs=3, validation_data=test_generator)

# Image classification function
def classify_image(image_path, model, class_labels):
    img = Image.open(image_path).resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]
    print(f"Predicted Class: {predicted_class}")

# Text classification function
def text_classification():
    classifier = pipeline("text-classification")
    text = input("Enter a sentence for classification: ")
    result = classifier(text)
    print(f"Classification Result: {result}")
```

---

## Notes:
- Ensure the dataset is structured correctly before training.
- Install all required libraries such as `tensorflow` and `transformers` using pip.
