
import abc


class MatcherBase(object):
    __meta_class__ = abc.ABCMeta

    @abc.abstractmethod
    def similarity(self, scene_1, scene_2):
        pass
