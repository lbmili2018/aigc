from huggingface_hub import hf_hub_download
from ultralytics import YOLO

# path = hf_hub_download("Bingsu/adetailer", "face_yolov8n.pt")
path = "./face_yolov8n.pt"
model = YOLO(path)
import cv2
from PIL import Image

img = "/root/limiao/Gender/test/img/img_v2_b8625eed-0529-4dad-8190-52c4bed2a59g.jpg"
output = model(img)
print("output:", output[0].boxes.shape)
bbox = output[0].boxes.cpu()
print("bbox:", bbox)

bbox11 = bbox.boxes
print("bbox11:,",bbox11 )
x1 = float(bbox11[0,0])
y1 = float(bbox11[0,1])
x2 = float(bbox11[0,2])
y2 = float(bbox11[0,3])
print("x1, y1, x2, y2,", x1, y1, x2, y2)

# pred = output[0].plot()
# pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
# pred = Image.fromarray(pred)
# print("pred:", pred)

orig = Image.open(img)
face = orig.crop((x1, y1, x2, y2))
face.save("./txt2img_{}_{}.png".format(str(0), str(0)))