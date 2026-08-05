import os
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import easyocr


model = YOLO(r"D:\Smart Parking system\runs\detect\plate_detector-2\weights\best.pt")
reader = easyocr.Reader(['en'])


IMAGE_PATH = r"C:\Users\PC\Downloads\images_ocr\images (1).jpg"

results = model(IMAGE_PATH)
result_image = results[0].plot()

print(f" [Image: {filename}] ---> Detected Plate: {plate_number}")

plt.figure(figsize=(5, 5))
plt.imshow(result_image[:, :, ::-1])
plt.axis('off')
plt.title("Detection Result")
plt.show()


folder_path = r"C:\Users\PC\Downloads\images_ocr"

for filename in os.listdir(folder_path):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        full_path = os.path.join(folder_path, filename)
        img = cv2.imread(full_path)

        results = model.predict(img, conf=0.5)
        plate_number = "Not Detected"

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plate_crop = img[y1:y2, x1:x2]

            ocr_result = reader.readtext(plate_crop)
            if ocr_result:
                plate_number = " ".join([r[1] for r in ocr_result])

        print(f"{filename}: {plate_number}")
        

from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2

model = YOLO(r"D:\Smart Parking system\runs\detect\occupancy_detector_v2\weights\best.pt")

results = model.predict(source=r"C:\Users\PC\Downloads\images.jpg", conf=0.25, save=True)

result_image = results[0].plot()
result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(6, 6))
plt.imshow(result_image_rgb)
plt.axis('off')
plt.show()
