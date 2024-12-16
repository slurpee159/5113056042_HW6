
# README for HW6: Transfer Learning and Hugging Face Text Classification

## 1. Prompt: Implement Transfer Learning on VGG16 for Face Mask Detection

### Question:
How can I use VGG16 for transfer learning to classify images into `with_mask` and `without_mask` categories?

### Solution:
We use TensorFlow's `VGG16` pretrained model with custom classification layers for binary classification. The workflow involves building the model, preparing the dataset, training, and evaluating predictions.

#### Code Implementation:
```python
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models

# Build the model
vgg16_base = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
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

## 2. Prompt: Dataset Preparation and Splitting

### Question:
The dataset is not split into training and testing sets. How can I preprocess it?

### Solution:
We dynamically split the dataset into `train` and `test` directories, ensuring proper structure for TensorFlow's `ImageDataGenerator`.

#### Code Implementation:
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
        train_images, test_images = images[:split_idx], images[split_idx:]

        for img in train_images:
            shutil.copy(os.path.join(category_path, img), os.path.join(train_path, category, img))
        for img in test_images:
            shutil.copy(os.path.join(category_path, img), os.path.join(test_path, category, img))

    return train_path, test_path
```

---

## 3. Prompt: Train the Model and Evaluate Performance

### Question:
How can I train the VGG16 model and evaluate its accuracy?

### Solution:
We train the model using the prepared dataset and validate on the test set.

#### Code Implementation:
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

model.fit(train_generator, epochs=10, validation_data=test_generator)
```

---

## 4. Prompt: Image Classification

### Question:
How do I classify an image using the trained model?

### Solution:
We load an image from a URL or local path, preprocess it, and use the trained model to predict its class.

#### Code Implementation:
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

## 5. Prompt: Use Hugging Face for Text Classification

### Question:
How can I integrate Hugging Face's `pipeline` to classify text?

### Solution:
We use the `transformers` library to classify text using a pre-trained Hugging Face model.

#### Code Implementation:
```python
from transformers import pipeline

def text_classification():
    classifier = pipeline("text-classification")
    text = input("Enter a sentence for classification: ")
    result = classifier(text)
    print(f"Classification Result: {result}")
```

---

## Outputs

![image](https://github.com/user-attachments/assets/abf6e0cf-09c8-4414-956c-b664a28d0bde)

```

---



