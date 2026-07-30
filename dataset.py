from roboflow import Roboflow

rf = Roboflow(api_key= "IyWTII43oLEBOdS0ZmnY")

project = rf.workspace("mariano-hernandez-isdey").project("car-plate-detection-teqzn")
version = project.version(1)
dataset = version.download("yolov8")

print("Dataset downloaded to:", dataset.location)