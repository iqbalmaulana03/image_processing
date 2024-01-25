from flask import Flask, render_template, request
import cv2
import numpy as np

app = Flask(__name__)

def rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

def resize(image, width=416, height=416):
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

def gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def bilateralFilter(image, kernel_size=8, sig_color=15, sig_space=13):
    return cv2.bilateralFilter(image, kernel_size, sig_color, sig_space)

CLASS_MIN_AREA = 10

def object_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=136.0, tileGridSize=(25, 25))
    cl1 = clahe.apply(image)
    cl1 = 255 - cl1
    return cl1

def image_threshold_tozero(cl1, thres):
    _, thresh = cv2.threshold(cl1, thres, 255, cv2.THRESH_TOZERO)
    return thresh

def image_morphology(thresh):
    current = np.copy(thresh)
    prev = np.copy(current)
    prev[:] = 0

    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    current = cv2.morphologyEx(current, cv2.MORPH_OPEN, kernel5)
    iter_num = 0
    max_iter = 100

    while np.sum(current - prev) > 0 and iter_num < max_iter:
        iter_num = iter_num + 1
        prev = np.copy(current)
        current = cv2.dilate(current, kernel3)
        current[np.where(thresh == 0)] = 0

    return current

def filter_object_contour(current, image_bg_white, min_contour, max_contour):
    contours, _ = cv2.findContours(current, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    total_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_contour and area < max_contour:
            cv2.drawContours(image_bg_white, [contour], 0, [0, 255, 0], -1)
            total_area += 1
    return total_area, image_bg_white

def process_image(image_path):
    image = cv2.imread(image_path)
    image_rgb = rgb(image)
    image_resize = resize(image_rgb)
    image_gray = gray(image_resize)
    image_bilateral_filter = bilateralFilter(image_gray)
    image_clahe = object_clahe(image_bilateral_filter)

    threshold_value = 100
    image_threshold = image_threshold_tozero(image_clahe, threshold_value)
    image_morphology_result = image_morphology(image_threshold)

    min_contour_area = 10
    max_contour_area = 400

    total_area, result_image = filter_object_contour(
        image_morphology_result, image_clahe, min_contour_area, max_contour_area
    )

    print(f'Count area = {total_area}')
    classification = 'fertil' if total_area >= CLASS_MIN_AREA else 'infertil'

    result_image_resized = resize(result_image, 416, 416)
    result_image_resized = cv2.cvtColor(result_image_resized, cv2.COLOR_BGR2RGB)
    cv2.imwrite('static/fff.jpg', result_image_resized)
    
    return result_image_resized, classification

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_path = 'uploads/' + file.filename
            file.save(file_path)
            result_image, classification = process_image(file_path)
            return render_template('result.html', result_image=result_image, classification=classification)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
