from ultralytics import YOLO
import cv2

# -----------------------------------------------------------
# 1. Load your trained YOLOv11 model (.pt file)
# -----------------------------------------------------------
model = YOLO("./best.pt")     # <-- change this

# -----------------------------------------------------------
# 2. Run inference on your test image
# -----------------------------------------------------------
results = model("/content/chess_specimen.jpg")   # <-- change this

# -----------------------------------------------------------
# 3. Visualize predictions (bounding boxes + labels)
# -----------------------------------------------------------
# results[0].plot() returns a numpy image with drawings
annotated_img = results[0].plot()

# -----------------------------------------------------------
# 4. Save the image with predictions
# -----------------------------------------------------------
cv2.imwrite("predicted_output.jpg", annotated_img)
print("Saved prediction to predicted_output.jpg")

# -----------------------------------------------------------
# 5. (Optional) Show the output window
# -----------------------------------------------------------
cv2.imshow("YOLOv11 Prediction", annotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()