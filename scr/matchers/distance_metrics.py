import numpy as np
from sklearn.neighbors import DistanceMetric


class DistanceBuilder(object):
    def __init__(self, metric='euclidean', is_norm=True):
        self.dist = DistanceMetric.get_metric(metric)
        self.is_norm = is_norm

    def over_all_distance(self, feats):
        dist_metrics = self.dist.pairwise(feats, feats)
        dist_metrics = self.norm_dist(dist_metrics)

        return dist_metrics

    def norm_dist(self, dist_metrics):
        dist_metrics = (dist_metrics - np.min(dist_metrics)) / (np.max(dist_metrics) - np.min(dist_metrics))
        return dist_metrics
