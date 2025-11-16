# License-Plate-Localization
OpenCV project detects license plates by resizing, gray scaling, blurring, canny edge detection, and morphological closing. Finds all contours and selects the correct plate by scoring each contour’s internal edge density.

**Summary** 

Non-AI license plate detection system using OpenCV. The pipeline first standardizes images by resizing, gray scaling, and applying a Gaussian blur. A Canny edge detector is used to find edges, which are then connected into solid blobs using a morphological closing operation. A two-stage process is applied to the resulting contours . First, potential contours and filtered, and then based on the filtering, the best contour is chosen by identifying the one with the highest edge density (the ratio of edge pixels to total area) from the original canny image. Finally, the best contour bounding box is selected and cropped, providing a simple and efficient approach for license plate localization.


**Course concepts used**

1.Grayscale

2.Blurring

3.Edge detection

4.Morphological closing


**Additional concepts used**

1.Edge density

2.Contour filtering


**Dataset**

https://www.kaggle.com/datasets/fareselmenshawii/large-license-plate-dataset


**Novelty** 

•	Uses the original Canny edge detection as a data source, introducing edge density (the ratio of edge pixels to total area) as the key scoring metric, thereby assigning scores to each contour. 

•	Operating on the assumption that a real license plate, being full of characters, will have the highest edge density of all the candidates, allowing it to be distinguished from cleaner noise like car grilles or bumpers.


**Contributors**

1.Shikha Chikgouda (PES1UG23EC284)
   
2.Shreya S Naik (PES1UG23EC290)

3.Supriya Prasanna Kumar (PES1UG23EC315)


**References**

https://github.com/MYoussef885/License-Plate-Recognition

https://www.geeksforgeeks.org/python/python-opencv-morphological-operations/

https://www.geeksforgeeks.org/python/find-and-draw-contours-using-opencv-python/

https://stackoverflow.com/questions/40203932/drawing-a-rectangle-around-all-contours-in-opencv-python


**Limitations and Future Work**

•	Reliance on hard-coded filters for area, aspect ratio, and edge density, which might fail on some images with different scales, angles, or poor lighting.

•	The most logical future work is to add an OCR (Optical Character Recognition) engine, like Tesseract, to read the text from the cropped plates.

•	The entire pipeline could be replaced by a trained AI object detection model (like YOLO) to solve all these limitations and achieve true accuracy.






