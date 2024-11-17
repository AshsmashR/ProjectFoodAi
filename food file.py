
#STEP 2 CREATE DATASET WITH THE CORRESPONDING FOOD ITEMS CALORIES AND PER SERVINGS

import pandas as pd
data = {
    'Food Item': ['Burger', 'Butter Naan', 'Chai', 'Chapati', 'Chole Bhature', 'Dal Makhani', 'Dhokla', 'Fried Rice', 'Idli', 'Jalebi', 'Kaathi Rolls', 'Kadai Paneer', 'Kulfi', 'Masala Dosa', 'Momos', 'Paani Puri', 'Pakode', 'Pav Bhaji', 'Pizza', 'Samosa'],
    'Class Index': list(range(20)),
    'Approximate Calories per Serving': ['250–500', '150–200', '100–120', '70–100', '450–600', '300–350', '150–200', '250–300', '35–50', '150–200', '200–250', '300–400', '150–200', '250–300', '35–50', '150–200', '150–200', '400–500', '200–300', '150–200'],
    'Serving Size Description': ['1 medium-sized burger', '1 piece', '1 cup (240 ml)', '1 piece', '1 plate (2 bhature with chole)', '1 cup', '2 pieces', '1 cup', '1 piece', '2 pieces', '1 roll', '1 cup', '1 stick', '1 piece', '1 piece', '6 pieces', '2 pieces', '1 plate', '1 slice', '1 piece']
}
df = pd.DataFrame(data)
df.to_csv('food_calories.csv', index=False)
print("Excel file 'food_calories.csv' has been created.")
#LOCATE YOUR FILE


df= pd.read_csv(r'food_calories.csv')
print(df)
import pandas as pd
food_calorie_dict = df.set_index('Food Item')[['Approximate Calories per Serving', 'Serving Size Description']].to_dict('index')
print(food_calorie_dict)
import json
with open('food_calorie_info.json', 'w') as json_file:
    json.dump(food_calorie_dict, json_file)
print("Calorie dictionary saved as 'food_calorie_info.json'.")
#DICT IS MADE AND SAVED



from tensorflow.keras.models import load_model      #type:ignore
cnn_model = load_model('food_classification_model.keras')
print("Model loaded successfully.")

from tensorflow.keras.preprocessing.image import load_img, img_to_array      #type:ignore
import numpy as np
import matplotlib.pyplot as plt
def predict_with_calorie_info(image_path, model, class_indices, calorie_dict):
    img = load_img(image_path, target_size=(255, 255))  
    img_array = img_to_array(img) / 255.0  
    img_array = img_array[np.newaxis, ...]


    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    
    
    class_labels = {v: k for k, v in class_indices.items()}  # Reverse mapping
    predicted_class = class_labels[predicted_class_index]
    confidence = predictions[0][predicted_class_index] * 100

    calorie_data = calorie_dict.get(predicted_class, {
        'Approximate Calories per Serving': 'N/A',
        'Serving Size Description': 'N/A'
    })

    plt.imshow(load_img(image_path))
    plt.axis('off')
    plt.title(
        f"Predicted: {predicted_class}\n"
        f"Confidence: {confidence:.2f}%\n"
        f"Calories: {calorie_data['Approximate Calories per Serving']} kcal\n"
        f"Serving: {calorie_data['Serving Size Description']}",
        fontsize=14
    )
    plt.show()

    return predicted_class, confidence, calorie_data




with open('food_calorie_info.json', 'r') as json_file:
    food_calorie_dict = json.load(json_file)
print("Calorie dictionary loaded successfully.")

#NOW LETS TEST WITH SINGLE IMAGES BECAUSE OF LOW POWER LAPTOP
test_image_path = r"E:\dataset\Dataset\val\pizza\012.jpg" 
predicted_class, confidence, calorie_data = predict_with_calorie_info(
    test_image_path, cnn_model, train_generator.class_indices, food_calorie_dict
)
print(f"Predicted Class: {predicted_class}")
print(f"Confidence: {confidence:.2f}%")
print(f"Calories: {calorie_data['Approximate Calories per Serving']} kcal")
print(f"Serving Size: {calorie_data['Serving Size Description']}")
print(f"Predicted Class: {predicted_class}")
print("Available keys in food_calorie_dict:")
print(food_calorie_dict.keys())
print("Class names in train_generator.class_indices:")
print(list(train_generator.class_indices.keys()))


# MY DICT WAS NOT WORKING SUCCESSFULLY  IN MAPPING 

#SUCCESSFUL CODE BELOW
#STANDARDIZE DICT TO MATCH THE NAMES OF FOOD IN THE IMG DATASET 

food_calorie_dict = {
    key.replace(' ', '_').lower(): value for key, value in food_calorie_dict.items()
}
print(food_calorie_dict.keys())


#DEF
def predict_with_calorie_info(image_path, model, class_indices, calorie_dict):
    # Load and preprocess the image
    img = load_img(image_path, target_size=(255, 255))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict the class
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    class_labels = {v: k for k, v in class_indices.items()}  # Reverse mapping
    predicted_class = class_labels[predicted_class_index]  # No need for additional formatting

    # Fetch calorie and serving size info
    calorie_data = calorie_dict.get(predicted_class, {
        'Approximate Calories per Serving': 'N/A',
        'Serving Size Description': 'N/A'
    })

    # Display the result with the image
    plt.imshow(load_img(image_path))
    plt.axis('off')
    plt.title(
        f"Predicted: {predicted_class.replace('_', ' ').capitalize()}\n"
        f"Confidence: {predictions[0][predicted_class_index] * 100:.2f}%\n"
        f"Calories: {calorie_data['Approximate Calories per Serving']} kcal\n"
        f"Serving: {calorie_data['Serving Size Description']}",
        fontsize=14
    )
    plt.show()

    return predicted_class, predictions[0][predicted_class_index] * 100, calorie_data


test_image_path = r"E:\-Burgers-Chicken.jpg"

predicted_class, confidence, calorie_data = predict_with_calorie_info(
    test_image_path, cnn_model, train_generator.class_indices, food_calorie_dict
)

print(f"Predicted Class: {predicted_class.replace('_', ' ').capitalize()}")
print(f"Confidence: {confidence:.2f}%")
print(f"Calories: {calorie_data['Approximate Calories per Serving']} kcal")
print(f"Serving Size: {calorie_data['Serving Size Description']}")

#THE MEASURMENTS ARE NOT ABSOLUTE OR CONSTANT. SUBJECT TO CHANGES AS I WILL UPDATE MY DATASET 
#TO INCLUDE MORE FOOD AND IMPROVE CALORIE ESTIMATIONS