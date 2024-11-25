# Custom YOLOv5 for Car Detection using COCO Dataset
This project is a custom-trained YOLOv5 model for car detection using transfer learning with the first 10 layers frozen, trained on the COCO 2017 dataset. It includes scripts for converting the dataset annotations to YOLO format, training the model, and evaluating results on test images. This project is licensed under the GNU General Public License v3.0 (GPLv3).
### Progess Tracking ( Project Milestone)
I am also still working on this repo and notebooks to make it modular to use as you wish, and just need to change some code and it will work (but for now it is not complete)  
I have uploaed all my result from my training in the folder exp8, This folder contains my trainined model and all the metric results from the training and validation  
As of now my `project_milestone.ipynb` was run and some test inference on validation images and test images and they show up.  
### For now if you try to run this file. it will not work since I do not have the data set in this repo since it is big. But towrards the end of the file all my result and  inference testing is shown.  
Apart from that you can take a look at all the result serperatly in the exp8 folder and my weight/model in the weights folder
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
- Clone yolov5 into the project at the root level
  `git clone https://github.com/ultralytics/yolov5`
- Required Python libraries:
  These needs be run in the yolov5 directory
  - `pip install -r requirements.txt`
  This can be found in the requierment .txt in the yolov5 folder
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
6. Train the Model
   Run the following command to train the YOLOv5 model with tranfer learning and freeze the first 10 layers:
   Note: You have to download the pretrained model speratly, I am using the yolo5s.pt (small)
   ```
   python train.py --img 640 --batch 32 --epochs 10 --data coco_car.yaml --weights yolov5s.pt --freeze 10

   ```
7. Test the Model
   To evaluate the model's performance on the COCO test images, run the following command
   ```
   python detect.py --weights yolov5/runs/train/exp/weights/best.pt --img 640 --conf 0.25 --source data/coco/images/test2017 --project runs/detect --name test_results --exist-ok

   ```
7. View Results
   - Training metrics (e.g., mAP, percision, recall) are logged in the `yolov5/runs/train/exp`
   - Detection results for test images will be saved in `yolov5/runs/test_results`

#### Directory Structure

```
project/
│
├── project_milestone.ipynb          # Jupyter Notebook
├── coco_car.yaml           # Dataset configuration for YOLOv5
├── data/
│   ├── coco/
│       ├── images/         # Train and validation images
│       ├── labels/         # YOLO-format annotations
│
├── yolov5/                 # YOLOv5 directory
│
└── .gitignore              # To exclude unnecessary files

```
---
# License
This project is licensed under the GNU General Public License v3.0 (GPL-3.0).

# Attribution:
- [YOLOv5](https://github.com/ultralytics/yolov5) is used under the AGPL-3.0 license.
- [COCO Dataset](https://cocodataset.org) is used under the CC BY 4.0 license.

You are free to use, modify, and distribute this project under the terms of the GPL-3.0 license.

