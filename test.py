
from roboflow import Roboflow
rf = Roboflow(api_key="hgZTHqf11q3G2r7uDxvW")
project = rf.workspace().project("weapon-detection-f1lih")
model = project.version(1).model

# infer on a local image
print(model.predict("guntest.jpg", confidence=40, overlap=30).json())

# visualize your prediction
# model.predict("your_image.jpg", confidence=40, overlap=30).save("prediction.jpg")

# infer on an image hosted elsewhere
# print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())