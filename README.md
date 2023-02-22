# Real-Time mechanical characterisation of microcapsules

In a recent research paper, entitled "A method for real-time mechanical characterisation of microcapsules" in Biomechanics and Modelling in Mechanobiology, we developped a novel multilayer perceptron (MLP)-based machine learning (ML) approach, for real-time simultaneous predictions of the membrane mechanical law type, shear and area-dilatation moduli of microcapsules, from their camera-recorded steady profiles in tube flow. Here we deposite the codes for image processing and MLP-based prediction, and the testing data in this repository.

## Structure of the codes

The codes are in the file "Real-Time_Capsule_Mechanics.cpp" within the folder "Real-Time_Capsule_Mechanics". 

The codes have two sequential parts: 

1. **Image processing**    The method for image processing has been described in detail in Section 3.3 of the paper. It is implemented with C++ using the OpenCV library (v4.7). Firstly, the camera-recorded image of a deformed capsule in a channel is subtracted with its background to remove the geometry of the tube and static noise. The image is then binarized and the capsule is centered to a square region of interest (RoI) with the width equal to the tube diameter. Secondly, the boundary of the capsule in the binary image is detected and the boundary nodes are sorted counterclockwisely. To reduce pixel noise, a piecewise second-order polynomial function is employed to approximate the capsule boundary smoothly, and the membrane nodes with equal arc-length distance are sampled from the fitted curve. Finally, coordinates of the membrane nodes are built into a 1D vector which is the input of the MLP model.

2. **MLP**    The approach can be found from Section 3.2 of the paper. We first develop and train the MLP using the open-source framework Tensorflow v2.5, based on Python. To speed up computation, we download the parameters of the trained MLP (stored in files "lawmodel.json", "camodel.json", "cmodel.json" and "pmodel.json") and reimplement the network using C++.

Details regarding the functions and varialbles of the codes can be found from the comments in the source-code file.

## How to use the codes

### Dependency

The codes requires the following libraries:

OpenCV 4.x          https://github.com/opencv/opencv

Eigen               https://eigen.tuxfamily.org/index.php?title=Main_Page

frugally-deep       https://github.com/Dobiasd/frugally-deep

You need to install OpenCV 4.x to your computer. The other two libraries are header only, and thus you don't need to intall them. 

### Run the codes

A VisualStudio project entitled "Real-Time_Capsule_Mechanics.sln" has been configured and can be found in the root folder. It can be used directly by interested users. Note that the project should be compiled with the "release mode" which is considerable faster than the "debug mode".  

To run the codes we provided two examples in the folder "Real-Time_Capsule_Mechanics\Testcases". More details of the data can be found from the section below.  In general the codes need two images as inputs: image "00000.png" is the background of the tube flow, and "00000.png" is the steady deformed capsule. As described in the previous section, the program will read both images, process them to get the 1D boundary of the deformed capsule, and predict the capsule mechanical properties.  

In principle the method is not limited to certain camera types or image resolutions. However, one should ensure that the two images mentioned above are visualising the same field and have the same resolution. 

## Testing data

In the folder "Real-Time_Capsule_Mechanics\Testcases" one can find the experimental data which we have used to test the accuracy and latency of the present method (in Section 4.2 of the paper). The experiments were conducted by Risso et al. (2006, JFM, https://doi.org/10.1017/S0022112005007652), where bioartificial capsules with a human serum albumin-alginate membrane were used. 


