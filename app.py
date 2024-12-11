import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras import Sequential, layers, models
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
)
from tensorflow import keras
from sklearn.metrics import accuracy_score
from tensorflow.keras.layers import RandomFlip, RandomRotation, Resizing, Rescaling
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Lambda
import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
import numpy as np
import joblib
import pickle

st.title("FRUITS AND VEGETABLE FRESHNESS DETECTION WEB APP")

frame_placeholder = st.empty()
freshness_result_placeholder = st.empty()

if "run_camera" not in st.session_state:
    st.session_state.run_camera = False

col1, col2 = st.columns(2)
with col1:
    if st.button("Start Camera"):
        st.session_state.run_camera = True

with col2:
    if st.button("Stop Camera"):
        st.session_state.run_camera = False

def rgb_to_grayscale(image):
    return tf.image.rgb_to_grayscale(image)

@st.cache_resource
def load_models():
    classification_model = loaded_model = tf.keras.models.load_model(r"D:\work\Flipkart Grid\final\Freshness Calculator\models\model.h5", custom_objects={'rgb_to_grayscale': rgb_to_grayscale})
    label_encoder = joblib.load(r"D:\work\Flipkart Grid\final\Freshness Calculator\models\label_encoder.pkl")
    with open(r"D:\work\Flipkart Grid\final\Freshness Calculator\models\regression_model.pkl", "rb") as f:
        regression_model = pickle.load(f)
    return label_encoder, regression_model, classification_model
    
label_encoder, regression_model, classification_model = load_models()

def calculate_color_score(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, saturation, _ = cv2.split(hsv_image)
    mean_saturation = np.mean(saturation)
    color_score = (mean_saturation / 255) * 100
    return color_score

def calculate_shape_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return 0
    largest_contour = max(contours, key=cv2.contourArea)
    contour_area = cv2.contourArea(largest_contour)
    x, y, w, h = cv2.boundingRect(largest_contour)
    bounding_box_area = w * h
    extent = contour_area / bounding_box_area if bounding_box_area > 0 else 0
    shape_score = extent * 100
    return shape_score

def calculate_wrinkle_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 100)
    edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1])
    wrinkle_score = (1 - edge_density) * 100
    return wrinkle_score

def calculate_marks_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
    total_pixels = image.shape[0] * image.shape[1]
    dark_pixels = np.sum(thresh == 255)
    dark_ratio = dark_pixels / total_pixels
    mark_score = (1 - dark_ratio) * 100
    return mark_score

def extract_features(image):
    image = cv2.resize(image, (224, 244))
    color_score = calculate_color_score(image)
    shape_score = calculate_shape_score(image)
    wrinkle_score = calculate_wrinkle_score(image)
    mark_score = calculate_marks_score(image)
    return {
        "color_score": color_score,
        "shape_score": shape_score,
        "wrinkle_score": wrinkle_score,
        "mark_score": mark_score
    }

def freshness_calc(image):
    classes = ['Apple', 'Banana', 'Bittergourd', 'Capsicum', 'Cucumber', 'Okra', 'Oranges', 'Peaches', 'Pomergranate', 'Potato', 'Strawberry', 'Tomato']

    image_resized = cv2.resize(image, (128, 128))

    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

    image_array = image_rgb.astype('int')

    # Add batch dimension to the image array
    image_array = np.expand_dims(image_array, axis=0)

    predictions = classification_model.predict(image_array)
    predicted_class = classes[np.argmax(predictions)]

    print(predicted_class)
    features = extract_features(image)
    encoded_item = label_encoder.transform([predicted_class])[0] 

    features = list(features.values())
    features.append(encoded_item)

    freshness_score = regression_model.predict([features]) 
    return freshness_score, predicted_class

def draw_bbox_without_labels(frame, bbox):
    for box in bbox:
        x1, y1, x2, y2 = box
        # Draw the bounding box (rectangle) with a specific color (e.g., green) and thickness
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return frame

if st.session_state.run_camera:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Error: Could not open webcam.")
        st.session_state.run_camera = False
    else:
        try:
            while st.session_state.run_camera:
                ret, frame = cap.read()
                if not ret or frame is None:
                    st.error("Failed to capture frame.")
                    break

                # Detect objects
                bbox, labels, confidences = cv.detect_common_objects(frame)

                output_image = draw_bbox_without_labels(frame, bbox)

                cropped_objects = []
                for box in bbox:
                    x1, y1, x2, y2 = box
                    try:
                        cropped_object = frame[y1:y2, x1:x2]
                        if cropped_object.size != 0:
                            cropped_objects.append(cropped_object)
                        else:
                            print("Warning: Cropped object is empty.")
                    except Exception as e:
                        print(f"Error while cropping object: {e}")

                output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(output_image_rgb, caption="Real-Time Object Detection", use_container_width=True)

                # Display freshness score for each detected object
                for idx, obj in enumerate(cropped_objects):
                    try:
                        freshness_score, item_name = freshness_calc(obj)
                        if freshness_score:
                            freshness_result_placeholder.write(f"Freshness Score of {item_name}: {freshness_score}")
                        else:
                            freshness_result_placeholder.write(f"NA for {labels[idx]}")
                    except Exception as e:
                        freshness_result_placeholder.write(f"Error during Freshness Score Calculation: {e}")

                if not st.session_state.run_camera:
                    cap.release()
                    break

        except Exception as e:
            st.error(f"Error during processing: {e}")