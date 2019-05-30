# Bayesian Grasp： Vision based robotic stable grasp via prior tactile knowledge learning

This is a Pytorch implementation of learning prior tactile knowledge to achieve stable grasp. Details are described in the short paper Bayesian Grasp： Vision based robotic stable grasp via prior tactile knowledge learning.

## Getting Started

### Video demo

The video demo is uploaded to : https://www.youtube.com/watch?v=9Xe6YzANTQo&t=9s

### Prerequisites

python 2.7

Pytorch 1.0.1

### Installation

This implementation requires the following dependencies (tested on Ubuntu 16.04.4 LTS):

- Python 2.7 or Python 3

- [NumPy](http://www.numpy.org/), [SciPy](https://www.scipy.org/scipylib/index.html), [OpenCV-Python](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_tutorials.html), [Matplotlib](https://matplotlib.org/). You can quickly install/update these dependencies by running the following (replace `pip` with `pip3` for Python 3):

  ```
  pip install numpy scipy opencv-python matplotlib
  ```

- [PyTorch](http://pytorch.org/) version 1.0.1. See installation instructions [here](http://pytorch.org/previous-versions/) or run the following:

  ```
  pip install torch==1.0.1 torchvision==0.2.2
  ```

### Hardware Setup

![1558702526028](/home/schortenger/.config/Typora/typora-user-images/1558702526028.png)

​	As the figure shows, in this experiment, we use a hardware setup consisting of a 6-DoF UR5 arm from Universal Robots, the 2-finger adaptive robot gripper from Robotiq, Realsense D435 RGB-D camera and a TakkStrip produced by Takktile. Among them, the camera is mounted on the top of the desk to provide visual data. The TakkStrip is a tactile sensor which can measure pressures of 5 contact points on the surface simultaneously. It consists of 5 sensors end-to-end spaced at 8mm spacing, along with a "traffic cop" microchip that allows the entire strip to be accessed over I2C without address conflicts.

​	The network training is implemented in PyTorch on a server, consisting of a 4.2 GHz Intel Core i7-7700HQ CPU (four physical cores), 16GB of system memory, and a GTX 1080Ti GP.

### Dataset

The dataset is collected after 1500 grasps with 5 objects, and the dataset is uploaded:

### Model

traindataset.py is the file to train VGG16-Net and ResNet-50

```
cd tactile_prior
python traindataset.py
```

Bayesiangrasp.py is the file to implement stable grasp based on trained network

```
python Bayesiangrasp.py
```

