#TO DO MULTICLASSIFICATION OF FOOD IMAGES AND CALORIE ESTIMATION
#LETS GET SOME IMPORTS
#TYPED: ignore to supress intellisense warnings

import tensorflow as tf
from PIL import Image
import tensorflow.keras.backend as K #type: ignore
from tensorflow.keras.preprocessing import image #type: ignore
from tensorflow.keras import models #type: ignore
from tensorflow.keras.models import load_model #type: ignore
from skimage.io import imread
K.clear_session()
import cv2
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
import warnings
warnings.filterwarnings("ignore")

#LETS SEE SOME IMAGES AND THEIR DIMENSION IN NUMPY&MATPLT
image = plt.imread(r"E:\dataset\Dataset\train\chai\002.jpg")
dims = np.shape(image)
dims
matrix = np.reshape(image , (dims[0] * dims[1] , dims[2]))
print(np.shape(matrix))
plt.imshow(image)
print("Image Shape:-" , dims[:2])
print("Color Channels:-", dims[2])
print("Min Color Depth : {}, Max Color Depth {}".format(np.min(image) , np.max(image)))
plt.show()


#LETS visualization and identify class imbalances
import os
train = r"E:\dataset\Dataset\train"
df = dict()
for i in os.listdir(train):
    sub_dir = os.path.join(train , i)
    count = len(os.listdir(sub_dir))
    df[i] = count
keys = df.keys()
values = df.values()
colors = ["blue" if x<= 150 else "red" for x in values]
fig , ax = plt.subplots(figsize = (12,8))
y_pos = np.arange(len(values))
plt.barh(y_pos , values , align = "center" , color=colors)
for i , v in enumerate(values):
    ax.text(v+1.4 , i-0.25 , str(v), color = colors[i])
ax.set_yticks(y_pos)
ax.set_yticklabels(keys)
ax.set_xlabel('Images',fontsize=16)
plt.xticks(color='black',fontsize=13)
plt.yticks(fontsize=13)
plt.show()  

train_folder = r"E:\dataset\Dataset\train"
images = []
for food_folder in sorted(os.listdir(train_folder)):
    food_folder_path = os.path.join(train_folder, food_folder)
    
    if os.path.isdir(food_folder_path):
        food_items = os.listdir(food_folder_path)
        
        if food_items:
            food_selected = np.random.choice(food_items)
            images.append(os.path.join(food_folder_path, food_selected))
if images:
    fig = plt.figure(1, figsize=(25, 25))
    
    for subplot, image_ in enumerate(images):
        category = image_.split(os.sep)[-2]
        imgs = plt.imread(image_)
        a, b, c = imgs.shape
        fig = plt.subplot(5, 4, subplot + 1)
        fig.set_title(category, pad=10, size=18)
        plt.imshow(imgs)
    
    plt.tight_layout()
    plt.show()
else:
    print("No images found in the specified folder.")


#IMAGE PREPROCESSING(SIMPLE WAY)
from tensorflow.keras.preprocessing.image import ImageDataGenerator #type: ignore
n_classes = 20
batch_size = 32
img_width , img_height = 255, 255
train_data_dir = r"E:\dataset\Dataset\train"
train_datagen = ImageDataGenerator(rescale = 1./255,
                                  shear_range = 0.2,
                                  zoom_range = 0.2,
                                  horizontal_flip = True)

val_data_dir = r"E:\dataset\Dataset\val"
val_datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

val_generator = val_datagen.flow_from_directory(
    val_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')


#LETS SEE THOSE PROCESSED IMAGES
data_batch, label_batch = next(train_generator)
num_images = 16  
fig, axes = plt.subplots(4, 4, figsize=(12, 12))
axes = axes.flatten()

for img, ax in zip(data_batch[:num_images], axes):
    # Display each image
    ax.imshow(img)
    ax.axis('off')  
plt.tight_layout()
plt.show()

#LETS SEE WHAT AND HOW CLASSES ARE PRESENT IN OUR TRAIN
class_ = train_generator.class_indices
class_


#LETS BUILD AND TRAIN MODEL (EXCITING PART)

import tensorflow as tf
from tensorflow.keras.models import Sequential #type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout #type: ignore
from tensorflow.keras.optimizers import Adam #type: ignore
from tensorflow.keras.regularizers import l2 #type: ignore


#LABEL THESE PARAMS FOR LESS CONFUSION
nb_train_samples = 3583
nb_valid_samples = 1089
img_height, img_width = 255, 255  
n_classes = 20  #WE HAVE 20 FOOD ITEMS AS OF NOW


#START MODEL
cnn_model = Sequential([
    # Convolutional layers
    Conv2D(16, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    # Flattening
    Flatten(),
    
    # Fully connected layers
    Dense(64, activation='relu'),
    Dropout(0.1),
    Dense(n_classes, kernel_regularizer=l2(0.005), activation='softmax')  # Output layer with 20 classes
])


cnn_model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
cnn_model.summary()

history = cnn_model.fit(train_generator,
                             steps_per_epoch = nb_train_samples // batch_size,
                             validation_data = val_generator,
                             validation_steps = nb_valid_samples // batch_size,
                             epochs = 10)

class_ = train_generator.class_indices
class_

val_loss, val_accuracy = cnn_model.evaluate(val_generator, steps=nb_valid_samples // batch_size)
print(f"Validation Accuracy: {val_accuracy:.2f}")
print(f"Validation Loss: {val_loss:.2f}")
#NOT VERY GOOD JOB BUT OKAY FOR MULTICLASSIFICATION


from tensorflow.keras.preprocessing.image import load_img, img_to_array #type: ignore
def predict_single_image(image_path, model, class_indices):
    img = load_img(image_path, target_size=(img_height, img_width))
    img_array = img_to_array(img) / 255.0  
    img_array = np.expand_dims(img_array, axis=0)  


    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)

    
    class_labels = {v: k for k, v in class_indices.items()}  
    predicted_class = class_labels[predicted_class_index]
    confidence = predictions[0][predicted_class_index] * 100

    return predicted_class, confidence

#LETS TEST ONE IMG
test_image_path = r"E:\images t.jpeg"# Replace with your test image path
predicted_class, confidence = predict_single_image(test_image_path, cnn_model, train_generator.class_indices)
print(f"Predicted Class: {predicted_class}")
print(f"Confidence: {confidence:.2f}%")


#BUT I WANT TO SEE IMAGE AND CONFIRM!!
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array #type:ignore
def predict_and_show_image(image_path, model, class_indices):
    img = load_img(image_path, target_size=(img_height, img_width))
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)

    class_labels = {v: k for k, v in class_indices.items()}  # Reverse mapping
    predicted_class = class_labels[predicted_class_index]
    confidence = predictions[0][predicted_class_index] * 100

#JUST ADD THE PLT FUNC TO SEE THEM!

    plt.imshow(load_img(image_path))
    plt.axis('off')
    plt.title(f"Predicted: {predicted_class}\nConfidence: {confidence:.2f}%", fontsize=14)
    plt.show()

    return predicted_class, confidence

test_image_path = r"E:\peporni.jpg"
predicted_class, confidence = predict_and_show_image(test_image_path, cnn_model, train_generator.class_indices)
print(f"Predicted Class: {predicted_class}")
print(f"Confidence: {confidence:.2f}%")



#SAVE THE BEAUTIFUL MODEL WHICH TOOK HOURS TO RUN ON CPU LOL
cnn_model.save('food_classification_model.keras')
print("Model saved as 'food_classification_model.keras'")
class_ = train_generator.class_indices
class_
#PRINTED CLASS_ AS I HAVE CREATE DICT KEYS TO MAP THE MODEL TO DO CALORIE ESTIMATION