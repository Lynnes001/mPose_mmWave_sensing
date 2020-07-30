# Capture 3D Skeleton Points from Xbox Kinect.

## Hardware Prerequisits

- Xbox Kinect v1
- Xbox Windows PC adapter

## Software Prerequisits

- Windows OS
- See [PyKinect2](https://github.com/Kinect/PyKinect2)

## Use


1. Run pykinect_example.py. It will give you the live streaming picture from Xbox Kinect overlapped with detected skeleton posture. It gives chance to visualize the skeleton output and find the best place to setup your device.

See [PyKinect2](https://github.com/Kinect/PyKinect2) for sample outputs.


2. Config pykinect_skeleton.py and run. This code captures a number of skeleton frames w/timestamp, and save as a npy file. Read the code for more details.