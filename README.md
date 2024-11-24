# Car Detection using YOLOv5
This repository contains a project focused on fine-tuning the YOLOv5 model for car detection using the COCO dataset. The model has been trained on car-specific annotations and demonstrates its performance through percision, recall, and mAP metrics

### Progess Tracking ( Project Milestone)
The `project_milestone.ipynb` serves as the primary reference for this project milestone. It contains:
- Dataset preparation steps
- Conversion of COCO to YOLO annotations
- Initial Training results
- Visualization of training metrics (loss, precision, recall, and mAP)
- Predications on validation and test images with visual feedback
To review my progress, open the notebook
#### Future Work
- Train the model for 5 more epochs to improve performace ( I have trained it on 10 right now )
- Deploy the trained model as a web app using Flask for FastAPI framework
- Explore adding other object categories for detection
---
#### Features
- **Model:** YOLOv5 (fine-tuned for car detection)
- **Dataset:** COCO dataset filtered for car annotations
- **Performance Metrics:** Percision, Recall, mAP
- **Interactive Visualization:** Validation results and predictions displayed for specific images in my project_milestone notebook
---
#### Repository Contents
- `project_milestone.ipynb`: Jupyter Notebook with the full training and evaluation workflow
- `coco_car.yaml`: Custom dataset configuration for YOLOv5
---
# Setup Instructions
#### Requirements
- Python >= 3.7
- CUDA-enabled GPU (optional but recommended for training)
- Required Python libraries:
  These needs be run in the yolov5 directory
  - `pip install -r requirements.txt`
#### Download and Prepare the Dataset
1. Download COCO Dataset
   - Vist the [COCO Dataset Download](https://cocodataset.org/#download)
   - Download the following files:
     - 2017 Train Images
     - 2017 Val Images
     - 2017 Test Images
     - 2017 Train/Val annotations
2. Unzip the Files
   Run the following commands to unzip the dataset into the appropriate directory structure:
   ```
   mkdir -p data/coco/images/train2017
   mkdir -p data/coco/images/val2017
   mkdir -p data/coco/images/test2017
   mkdir -p data/coco/annotations

   unzip train2017.zip -d data/coco/images/train2017
   unzip val2017.zip -d data/coco/images/val2017
   unzip test2017.zip -d data/coco/images/test2017
   unzip annotations_trainval2017.zip -d data/coco/annotations
   ```
3. Clean Up
   Delete the zip files after unziping to save space
   ```
   rm train2017.zip
   rm val2017.zip
   rm test2017.zip
   rm annotations_trainval2017.zip

   ```
5. Convert COCO Annotations to YOLO Format
   User the custom notebook `coco_to_yolo.ipynb` to filter and convert annotations for cars only
   - Open the Notebook and execute the cells to:
     - Convert COCO annotations to YOLO format
     - Focus only on the car category
     - Automatically create teh necessary label files

#### Directory Structure

```
project/
│
├── notebook.ipynb          # Jupyter Notebook
├── coco_car.yaml           # Dataset configuration for YOLOv5
├── runs/                   # Training results and weights
├── data/
│   ├── coco/
│       ├── images/         # Train and validation images
│       ├── labels/         # YOLO-format annotations
│
├── yolov5/                 # YOLOv5 directory
│
└── .gitignore              # To exclude unnecessary files

```
