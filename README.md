
Problem Statement
Electronic waste (E-waste) is one of the fastest-growing waste streams globally, containing hazardous materials that can harm the environment if not disposed of properly.
Manual sorting of E-waste from general waste is time-consuming, inefficient, and prone to errors, making recycling and responsible disposal difficult, especially at small-scale collection centers.

Solution
We developed an embedded AI-based E-waste detection system using a Raspberry Pi and a custom YOLOv8 model trained on a dataset of E-waste and non-E-waste images.
Key Features of the Solution:
-Real-time Detection: Uses a live camera feed to identify items as “E-waste” or “Non E-waste.”
-Embedded Deployment: Runs efficiently on Raspberry Pi, demonstrating hardware-software integration.
-Custom Dataset: Trained on images collected and labeled via Roboflow, tailored for real-world E-waste scenarios.
-Scalable & Portable: Can be extended to sorting mechanisms (servo motors, bins) for automated separation.
-Visualization: Shows bounding boxes and confidence scores for each detected item, providing instant feedback.

Hardware Requirements
Raspberry Pi 4 model B (4GB)
Pi Camera Module v2 or USB camera
MicroSD card (32GB+)
Power supply for Raspberry Pi

Software Requirements
Raspberry Pi OS (64-bit recommended)
Python 3.x
Libraries: torch, ultralytics, opencv-python, numpy

Raspberry Pi OS Installation:
Download the Raspberry Pi Imager from the official Raspberry Pi website
Insert your microSD card into your PC.
Open Raspberry Pi Imager → Select OS → Choose Raspberry Pi OS (64-bit).
Select the SD card → Click Write.
Insert SD card into Raspberry Pi → Connect peripherals → Power on.

Complete initial setup:
Set username/password
Connect to Wi-Fi

Update system:
sudo apt update && sudo apt upgrade -y

Enable camera interface:
sudo raspi-config
Navigate: Interface Options → Camera → Enable
Reboot after enabling

Setup Python Environment

Install dependencies:
sudo apt install python3-pip
pip3 install torch opencv-python ultralytics numpy

Create a virtual environment:
python3 -m venv venv
source venv/bin/activate
Clone Repository & Upload Dataset

Clone this repository:
git clone <your-repo-link>
cd E-Waste-Detection

Add your custom Roboflow-trained dataset:
Dataset should contain images/ and labels/ folders.
Ensure the dataset classes are “e-waste” and “non e-waste”.

Run YOLOv8 Detection

Run on live camera feed:
python3 detect_e_waste.py

Press q to quit the feed.

Python script (detect_e_waste.py) sample:

from ultralytics import YOLO

# Load custom YOLOv8 model
model = YOLO('best.pt')

# Run live camera detection
model.predict(source=0, show=True)
## Live Video Demo
Watch a live demo showing E-waste vs Non E-waste detection:

[Click here to watch the demo](https://drive.google.com/file/d/1L95Q7rIs2tUt6-JCbC8OgfbiaedQkbn7/view?usp=sharing)
