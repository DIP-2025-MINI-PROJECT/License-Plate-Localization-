import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

input_folder='test_img'
output_folder='output_test_img'

output_bounding_box_folder=os.path.join(output_folder, 'bounding_box')
output_cropped_folder= os.path.join(output_folder, 'cropped')
os.makedirs(output_bounding_box_folder, exist_ok=True)
os.makedirs(output_cropped_folder, exist_ok=True)

image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff')
image_paths = []
for ext in image_extensions:
    image_paths.extend(glob.glob(os.path.join(input_folder, ext)))
    
for img_path in image_paths:
    base_filename = os.path.basename(img_path)
    name_only = os.path.splitext(base_filename)[0]
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"Warning: Could not read {img_path}. Skipping.")
        continue

    scale = 620 / img.shape[1]
    dim = (620, int(img.shape[0] * scale))
    img_resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(img_blur, 100, 200)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
    closed = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, k)
    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lic_plate = []
    img_bounding_box = img_resized.copy()
    scores = []
    
    for i in contours:
        min_rect = cv2.minAreaRect(i)
        (x, y), (width, height), angle = min_rect
        if width < height:
            width, height = height, width
            angle = (angle + 90) % 180
        aspect_ratio = width / height if height > 0 else 0
        area = cv2.contourArea(i)
        
        if (aspect_ratio > 1.5 and aspect_ratio < 5.5) and (area > 1000 and area < 40000):
            x_br, y_br, w_br, h_br = cv2.boundingRect(i)
            canny_crop = canny[y_br:y_br+h_br, x_br:x_br+w_br]
            
            if canny_crop.size > 0:
                edge_density = np.sum(canny_crop > 0) / canny_crop.size
                scores.append((edge_density, i, min_rect))
                
    if scores:
        scores.sort(key=lambda s: s[0], reverse=True)
        best_score, best_contour, best_rect = scores[0]
        box = cv2.boxPoints(best_rect)
        box = np.intp(box)
        cv2.drawContours(img_bounding_box, [box], 0, (0, 255, 0), 2)
        lic_plate.append(box)
        
    if lic_plate:
        detection_savename = os.path.join(output_bounding_box_folder, f"{name_only}_bounding_box.jpg")
        cv2.imwrite(detection_savename, img_bounding_box)
        
        for j, extr_plate  in enumerate(lic_plate):
            x, y, w, h = cv2.boundingRect(extr_plate)
            cropped_plate = img_resized[y:y+h, x:x+w]
            if cropped_plate.size > 0:
                cropped_savename = os.path.join(output_cropped_folder, f"{name_only}_cropped.jpg")
                cv2.imwrite(cropped_savename, cropped_plate)