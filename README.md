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
[[ 0.         31.05033882 33.02158539 48.19921677]
 [31.05033882  0.         29.08769501 47.7989116 ]
 [33.02158539 29.08769501  0.         47.80762468]
 [48.19921677 47.7989116  47.80762468  0.        ]]
'''

# Get similar indexes
similar_indexes = src_serving.filter(dist_metrics,thresh = 0.3)
print(similar_indexes)
'''
Output:
[0, 1, 2]
'''
```
