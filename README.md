# scene_recognition
Implementation for scene matching or filtering library.

## Installation
```
pip3 install -r requirements.txt
python3 setup.py install
```

## Quick Start

```python
from scr.serving import SceneRecognitionServing

scene_1_path = 'img/scene_1.jpg'
scene_2_path = 'img/scene_2.jpg'
scene_3_path = 'img/scene_3.jpg'

scenes = [scene_1_path,scene_2_path,scene_3_path]

# create scene recognition serving
src_serving = SceneRecognitionServing()

# Get distance metrics
dist_metrics = src_serving.distance_metrics(scenes)
print(dist_metrics)
'''
Output:
[[1.00000000e+00 3.50396530e-01 3.09282032e-01 1.11022302e-16]
 [3.55791631e-01 1.00000000e+00 3.91567868e-01 8.30522155e-03]
 [3.14893735e-01 3.91456959e-01 1.00000000e+00 8.12444936e-03]
 [1.11022302e-16 0.00000000e+00 1.11022302e-16 1.00000000e+00]]
'''

# Get similar indexes
similar_indexes = src_serving.filter(dist_metrics,thresh = 0.3)
print(similar_indexes)
'''
Output:
[0, 1, 2]
'''
```

## Dataset and Trained weights
* Download on https://drive.google.com/drive/folders/1uQHmlLky6z-aV4yWxI1nuv8LHBpJ4b4x?usp=sharing

## Experiments scripts example
```
python3 examples/evaluation.py
python3 examples/experiments_ploter.py

```
