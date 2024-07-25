Wound Detection using YOLOv8
============================

 

Overview
--------

This project contains a YOLOv8-based segmentation model which is designed to segment wounds and a reference image from input images.
The reference image have a known surface area which is then used to calculate the actual area of segmented wound from the image.


Features
--------

-   **Wound Detection and Segmentation:** Accurately detects and then segments wound in images.

-   **Reference Segmentation:** Accurately detects and then segments reference image to assist in
    wound area calculations.

-   **Custom Dataset:** Trained on manually labeled custom dataset for precise
    segmentation.

-   **Easy Retraining:** Easy to retrain on similar dataset for even more
    precise calculations..

 

Offline Installation
--------------------

To use the model for offline detection, follow these steps:

1.  Close the Repository

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ bash
git clone https://github.com/void-1409/wound_segmentation.git
cd wound_segmentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

2.  Install virtualenv package (if you don't have it already) and create a python virtual environment, then
    activate the environment.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ bash
pip install virtualenv
virtualenv venv
venv\Scripts\activate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

3.  Install required Dependencies

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ bash
pip install -r requirements.txt
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

4.  Run the segmentation

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ bash
yolo segment predict model=segment_v3.pt source="images/test"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Note: change the path of `source` in above line to your image-dataset to test.
And this is only for segmentation using the trained model.

5. Run python script to calculate wound area

~bash
python area_image.py
~

Note: Change the Parameters `path_img` and `output_path` in order to calculate the area of your custom image.

 

Dataset
-------

The custom dataset used for training this model consists of manually labeled
images, with two classes.

-   **Wound Class:** labelled as `wound`.

-   **Reference Class:** labelled as `reference`.

 

Results
-------

The results of training and detection of different models are stored in `runs/segment`
directory, with bounding box and mask drawn on the segmented wounds and reference
images.

 

Here is one example of segmentation of an image from the test dataset:

![Wound and Reference segmentation of a test
image](./runs/segment/predict3/JOSE EMILIOIMG1435.jpg)

 

Also, here are the results from final training:

![Training Results](./runs/detect/train/results.png)

 

Retraining the Model
--------------------

In order to retrain the model, you can use pre-trained weights `best.pt` from train1 or train3 model.
The first training, train1, was done only on wound class and there was no reference class.
The last training, train3, was done on top of first training with both wound and reference classes in training set.
Further information for training a YOLOv8 segmentation model can be found on Ultralytics website [here](https://docs.ultralytics.com/tasks/segment)

 

 
