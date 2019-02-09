from scr.serving import SceneRecognitionServing

scene_1_path = 'img/scene_1.jpg'
scene_2_path = 'img/scene_2.jpg'
scene_3_path = 'img/scene_3.jpg'
scene_4_path = 'img/scene_4.jpg'

scenes = [scene_1_path, scene_2_path, scene_3_path, scene_4_path]

# create scene recognition serving
src_serving = SceneRecognitionServing()

# Get distance metrics
dist_metrics = src_serving.distance_metrics(scenes)
print(dist_metrics)

# Split similar and not similar scene
similar_indexes = src_serving.filter(dist_metrics, thresh=0.31)
print(similar_indexes)
