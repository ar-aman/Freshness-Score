import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import joblib
import pickle

def calculate_color_score(image):
    
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _ , saturation ,_ = cv2.split(hsv_image)
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

    item_name = "Banana"

    features = extract_features(image)

    label_encoder = joblib.load(r"D:\work\Flipkart Grid\final\Freshness Calculator\models\label_encoder.pkl")

    encoded_item = label_encoder.transform([item_name])[0]

    features = list(features.values())

    features.append(encoded_item)

    with open(r'D:\work\Flipkart Grid\final\Freshness Calculator\models\regression_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    freshness = model.predict([features])
    
    return freshness

cap = cv2.VideoCapture(0)

try:
    while True:
        try:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Failed to capture frame. Retrying...")
                continue

            try:
                bbox, labels, confidences = cv.detect_common_objects(frame)
            except Exception as e:
                print(f"Error during object detection: {e}")
                continue

            output_image = draw_bbox(frame, bbox, labels, confidences)

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

            for idx, obj in enumerate(cropped_objects):
                try:
                    freshness_score = freshness_calc(obj)

                    if freshness_score:
                        print(f"Freshness Score of object {idx + 1}: {freshness_score}")
                    else:
                        print(f"NA for {idx + 1}")

                except Exception as e:
                    print(f"Error during Freshness Score Calculation: {e}")

            cv2.imshow("Object Detection", output_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Terminating application...")
                break

        except KeyboardInterrupt:
            print("Interrupted by user. Exiting...")
            break

except Exception as e:
    print(f"Unexpected error: {e}")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Resources released. Windows closed.")

