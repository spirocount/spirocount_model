#Training a TF-Lite Object Dectection model with Spirochete Images

###Brief Summary
*Update: 11/28/2021*
This is a test to see if convloutional neural network running on a cell phone can be used for object detecrtion and counting of morphologically diverse bacteria (spirochetes). This project was inspired by this tutorial by the tensorflow team: 

* https://www.youtube.com/watch?v=vLxn5mOuWAk&t=1529s
* https://codelabs.developers.google.com/tflite-object-detection-android#0

###1. Environment

This project is currently using windows 10. Anaconda, CUDA, CuDNN are installed using this [tutorial] (https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10)

The requirements for the notebooks below are:
* python 3
* keras==2.5.0
* tflite-model-maker==0.3.1 (this should install tensorflow GPU for itself.)
* tensorflow==2.5.0

Installing in the order presented seemed to resolve potential problems. Additional packages openCV, numpy, pandas, os.

[Potentially useful compatibilitty list] 
(https://www.tensorflow.org/install/source#tested_build_configurations)

###2. Gethering Images and preprocessing

Images were taken under the darkfield microscope using a DSLR camera attached to the phototube.

(SHOW IMAGES HERE) 

The tensorflow_lite model maker recommends images sizes and formats. I used a OpenCV preprocess images JUPTR notebook here. For my training I selected and preprocessed images such that:
* images contained spirochetes that can be boardered by a box
* images are in jpg format
* images are square in dimension
* images are 960 by 960 resolution maximum 
* images were passed through google photo processing for contrast and brightness
* Each image was saved at each point in processing

Total of X images are use to build model in this repository. Images can be accessed here.

###3. Annotation

The objects in the image need to be annotated by hand. Annotation marks the location of the objects to be identified in each image. labelimg software was used for annotation.

https://github.com/tzutalin/labelImg

Rules for annotation I used
* sqare must have atleast one tail encompassed
* minmum the size of sqare as much as possible 
* class 1: high resolution image with morphology apparent
* class 2: out of focus
* calss 3: low resolution images with no morphology
Later I can combine or separate these classes to see if there are any benefits for model training.

(SHOW SCREENSHOT OF labelimg)

###4. Generate dataloader csv for TF_Lite model maker

The labelimg software outputs .xml file for each annotated image. The .xml file contains the class and coordinates of the objects in the image. The uses the xml_to_csv.py by Dan Tran found [here.] (https://github.com/datitran/raccoon_dataset) Then the data in the csv file is converted to a csv file format expected for the data loader for the TF_Lite model maker. Final data format in the csv is expalined [here] (https://www.tensorflow.org/lite/tutorials/model_maker_object_detection)

###5. Run model builder

I followed the code labs found [here.] (https://www.tensorflow.org/lite/tutorials/model_maker_object_detection)

(SHOW CODE BLOCKS)

###6. Enter model to test app

The model was entered into a test app found here
We are currently build an app which uses spirochete detecting model to count spirochetes here:

(SHOW TEST IMAGES)

###7. Evaluate models. 

EfficientDet-Lite0 -- poor detection
EfficientDet-Lite1 -- poor detection
EfficientDet-Lite2 -- works
EfficientDet-Lite3 -- batch size need to be reduced to train on my hardware. No improvement against EfficientDet-Lite2

Currently: more images are reqired to train a better more general model for spiochete detection





