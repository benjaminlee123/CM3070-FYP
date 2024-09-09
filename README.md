# CM3070-FYP
YOLOv5-LSTM for Scene Understanding and Classification
This repository contains the implementation of various machine learning models designed for scene understanding and classification, including:

CNN-based models for supervised learning on the Intel Image Classification dataset.
Unsupervised learning models for clustering tasks on the ADE20K dataset.
A YOLOv5-LSTM hybrid model that combines object detection with temporal sequence modeling.
Table of Contents
Overview
Datasets
Installation
Usage
Model Details
Results
Contributing
License
Overview
The goal of this project is to develop robust machine learning models that can effectively classify and understand scenes, combining both spatial and temporal information:

Intel Image Classification Dataset: A dataset with images of different scene types (e.g., buildings, forest, glacier, mountain, sea, and street).
ADE20K Dataset: An outdoor scene dataset used for clustering tasks to explore unsupervised learning techniques.
YOLOv5-LSTM Model: A hybrid model combining YOLOv5 for object detection and LSTM for capturing temporal dependencies between sequential scenes.
Datasets
Intel Image Classification Dataset:
Contains 3000 images, each resized to 150x150 pixels.
Six classes: buildings, forest, glacier, mountain, sea, and street.
ADE20K Outdoors Dataset:
4999 images of diverse outdoor scenes.
Used for unsupervised learning tasks.
Installation
To run the code, you'll need to install the necessary dependencies:

Clone the repository:

bash
Copy code
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
Create a virtual environment (optional but recommended):

bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
Install dependencies:

bash
Copy code
pip install -r requirements.txt
The requirements.txt file should contain:

text
Copy code
numpy
tensorflow
keras
scikit-learn
matplotlib
ultralytics
Usage
Data Preparation:

Ensure the datasets are in the correct directory paths specified in the code.
Run data_preparation.py to preprocess the datasets:
bash
Copy code
python data_preparation.py
Training the Models:

To train the baseline CNN model for the Intel dataset, run:
bash
Copy code
python train_cnn.py
To perform clustering on the ADE20K dataset, run:
bash
Copy code
python unsupervised_clustering.py
To train the YOLOv5-LSTM model, run:
bash
Copy code
python train_yolov5_lstm.py
Evaluation:

Use the provided scripts (evaluate_cnn.py, evaluate_yolov5_lstm.py) to evaluate the models' performance on the test datasets.
