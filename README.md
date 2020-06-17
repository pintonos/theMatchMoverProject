# Programming Project Computer Vision

This repository holds the code-base from the UIBK [computer vision project](https://orawww.uibk.ac.at/public_prod/owa/lfuonline_lv.details?sem_id_in=20S&lvnr_id_in=703612). A detailed description of the project can be found on the course [website ](https://iis.uibk.ac.at/courses/2020s/703612) (accessible via the UIBK network only).

## Objective 

In the project, we apply the techniques learned in the associated lecture. Starting from a source video that shows a static scene from multiple angles, we estimate the 3-D world coordinates for points in each frame. Next, the recreated model is used to insert an artificial object into the scene and re-render the video frames including the object. As a result, we inserted the artificial objected while taking into account the geometrical consistency of the scene.

## Authors
- Andreas Peintner (1515339)
- Josef Gugglberger (1518418)
- Tobias Kupek (11828471)

## Prerequisites

Our code was developed for Python 3.8.

The following packages are required.
Although the code should also run on different versions, we exclusively guarantee the functionality for the listed versions.

- `numpy 1.18.4`
- `opencv-python 4.2.0.34`
- `opencv-contrib-python  4.2.0.34`
- `pandas 1.0.3`
- `scipy 1.4.1`

They can be easily checked and installed via
```
pip install -r requirements.txt
```


## Config

Input and output files can be configured via `src/default.config`

The default configuration is sufficient to run the code and will download the example video automatically.

## Demo

We provide a **Jupyter notebook**, to test the code without the need to install any dependencies:
[https://colab.research.google.com/drive/1iBBZnpQBInIY_H4Xb1wP-FdoRGKzCoxQ?usp=sharing](https://colab.research.google.com/drive/1iBBZnpQBInIY_H4Xb1wP-FdoRGKzCoxQ?usp=sharing)

To run the **quick demo**, simply call
```
python src/main.py
```

All intermediate results are already provided to enable a fast demo.

To re-run the **calibration procedure**, call
```
python src/camera_calibration.py
```

To re-run the **frame-by-frame keypoint analysis**, remove or rename the following files
- `keyframes.npy`
- `keyframe_pts.npy`
- `keyframe_idx.npy`

## Report

A full work report, including our methods and chosen parameters can be found at

TODO

### Used tutorials and sources
- https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
- https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_pose/py_pose.html
- https://stackoverflow.com/questions/22180923/how-to-place-object-in-video-with-opencv/22192565#22192565
- https://stackoverflow.com/questions/21997021/augmented-reality-openglopencv/21999980#21999980
- https://medium.com/@omar.ps16/stereo-3d-reconstruction-with-opencv-using-an-iphone-camera-part-ii-77754b58bfe0
- https://bitesofcode.wordpress.com/2017/09/12/augmented-reality-with-python-and-opencv-part-1/
- https://avisingh599.github.io/vision/monocular-vo/
- https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_features_harris/py_features_harris.html