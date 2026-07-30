from roboflow import Roboflow

rf = Roboflow(api_key="IyWTII43oLEBOdS0ZmnY")

project = rf.workspace("parkinglot-idl5k").project("parking-lot-yjk6x")
version = project.version(1)
dataset = version.download("yolov8")

print("Dataset downloaded to:", dataset.location)