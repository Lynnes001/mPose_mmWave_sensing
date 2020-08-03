# mPose mmWave Radar Sensing

## Overview

This is a project implementation of mmWave radar human posture skeletons sensing and some wheels. Much appreciate to TI, TI e2e forum, [PreSense Radar](https://github.com/PreSenseRadar/OpenRadar) team, [PyKinect2](https://github.com/Kinect/PyKinect2), and [Tensorflow Keras](https://github.com/keras-team/keras) along implementing this project. Please star me if you think this repository is helpful.

The project includes the following parts.

1. Radar signal processing starting from raw mmWave radar
2. Frame sync between mmWave radar frames and Kinect skeleton frames
3. Wheels

## Prerequisite

### Hardware

- TI AWR1642EVM-ODS mmWave radar
- TI DCA1000EVM Data capture card
- Xbox Kinect V2


### Software

- [TI mmWave Studio v2.01](https://software-dl.ti.com/ra-processors/esd/MMWAVE-STUDIO/latest/index_FDS.html)
- [TI mmWave SDK](https://www.ti.com/tool/MMWAVE-SDK)
- Download [data](https://drive.google.com/file/d/16AnJaEJpsRUfh3Qct37vUvPs4iKCM5G9/view?usp=sharing) and save to `./data/`

## Wheels

- [Capture 3D Skeleton Points from Xbox Kinect.](https://github.com/Lynnes001/mPose_mmWave_sensing/blob/master/docs/capture_kinect.md)
- [Extract Range-Elevation-Azimuth Heatmap from Raw mmWave Signals](https://github.com/Lynnes001/mPose_mmWave_sensing/blob/master/docs/extractRangeAzimuthElevation.md)
- [Syncing radar frames and 3D skeleton from Kinect]



## References

- OpenRadar
- PyKinect2
- Keras