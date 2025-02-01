import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import numpy as np
import matplotlib.pyplot as plt

# Simulated Cobb angle calculation function (for demo purposes)
def cobb_angle_measurement():
    # Normally, you would use keypoint detection, Hough Transform, etc.
    # Here, we simulate this with a random Cobb angle for demonstration purposes.
    return np.random.uniform(0, 50)  # Simulate Cobb angle between 0 and 50 degrees

# CNN architecture for classification (as before)
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))  # Output: 3 categories (No, Mild, Severe)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Dummy prediction for demonstration (replace with actual trained model inference)
def predict_image(image_path):
    # Simulated prediction probabilities (use model.predict for real data)
    prediction_probs = np.array([0.2, 0.5, 0.3])  # Example output for Mild Scoliosis
    categories = ['No Scoliosis', 'Mild Scoliosis', 'Severe Scoliosis']
    predicted_class = categories[np.argmax(prediction_probs)]
    
    return predicted_class

# Show image with scoliosis classification and Cobb angle
def display_result(image_path):
    # Load the image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
    
    # Predict scoliosis category (dummy for now)
    scoliosis_category = predict_image(image_path)
    
    # Cobb angle measurement (simulated)
    cobb_angle = cobb_angle_measurement()
    
    # Determine scoliosis severity based on Cobb angle
    if cobb_angle < 10:
        scoliosis_severity = "No Scoliosis"
    elif 10 <= cobb_angle < 25:
        scoliosis_severity = "Mild Scoliosis"
    else:
        scoliosis_severity = "Severe Scoliosis"
    
    # Display the image along with the predicted category and Cobb angle
    plt.imshow(img)
    plt.title(f'Predicted: {scoliosis_category}\nCobb Angle: {cobb_angle:.2f}°\nScoliosis Severity: {scoliosis_severity}')
    plt.axis('off')  # Hide the axes for a cleaner output
    plt.show()

# Test the model on a new image
image_path = r'C:\Users\adith\transcriber_env\N43,S,66,F_1003_2.jpg'  # Replace with your test image path
display_result(image_path)