# Translation of AP Chest X-ray view into AP view

## Background and Motivation

Chest X-ray scan can be done in several views, which depends on the orientation of the patient relative to the X-ray machine during the procedure. Two the most popular views are posteroanterior (PA) and anteroposterior (AP). They differ in the direction how X-rays cross the body:

* In posteroanterior (PA) view the patient stands with his back to the X-ray machine. The rays penetrate into the patient's back and exit from the front of the body. Usually, pictures when the patient is standing are taken in this projection.
* In anteroposterior (AP) view the patient is rotated to the X-ray machine with his face. The rays penetrate into the front of the body and exit from the patient's back. There are some distortions in the images of this projection: the heart looks enlarged, the collarbones are often shifted upwards, etc. Therefore, these pictures need to be interpreted more carefully and the interpretation is harder. So, in this projection, pictures of bedridden patients are mainly taken: the X-ray machine is above the patient, and the patient is lying on his back.

To improve the quality, ease and speed up the interpretation of images in the AP projection, I propose to train a generative adversarial network that will translate images from the AP view to the PA view.

## Dataset limitations

Among the datasets devoted to chest radiography, there is none that has paired images in PA and AP projections of the same patient, because each X-ray scan is associated with the patient's irradiation. Therefore, I solve the problem of *unpaired* image to image translation. 

I use a dataset where images in the PA projection and in the AP projection are presented for different patients. I use the dataset from [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/data). 

I extract the view of the scan projection from the `ViewPosition` field of the metadata of the images presented in DICOM format. I have a balanced sample of

* 14204 scans in PA view;
* 11910 scans in AP view.


## Quality estimation

The quality of generating images is estimated via two classification networks:

* Classification into PA and AP view;
* Classification for the presence of pneumonia.

### Classification into PA and AP view

The model was trained on 99.44% accuracy (classes are balanced). The generated images in the PA projection (obtained from the AP projection) will be evaluated by this model. It is expected that perfectly generated images in PA projection will receive a prediction of PA with an accuracy of nearly 99%.


## References and readings

[radiologymasterclass.co.uk](https://www.radiologymasterclass.co.uk/tutorials/tutorials) A very cool website with short tutorials on understanding X-ray procedure and interpreting chest X-ray scans.