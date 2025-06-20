# Smart Traffic Management - Optimization and Analysis using YOLOv8

## Project Overview
1. The main aim of the proposed system is to develop a Smart Traffic Management System that uses YOLO based object detection to monitor and manage real-time traffic through CCTVs. 

2. The key features of the system include dynamic traffic signal control based on vehicle density and ambulance priority. 

3. The system is designed to improve traffic efficiency, ensure faster emergency response, and enhance overall road safety.


## Directory Structure
- `ambulance.py`, `ambulance1.py`, `ambulanceLlama.py`, `train_ambulance.py`: Main scripts for model training and inference.
- `util/`: Utility scripts and modules.
- `lanes/`: Example images for lanes and vehicles 
- `yolov8n.pt`: Pretrained model weights 
- `data.yaml`: Dataset configuration file.

## Setup
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place your datasets in the `train/`, `valid/`, and `test/` folders as per the structure.

## Usage
- To train the model:
  ```bash
  python train_ambulance.py
  ```
- To run inference:
  ```bash
  python ambulance.py
  ```

