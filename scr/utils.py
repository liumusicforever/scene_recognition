import os

from shutil import copy
from tqdm import tqdm


class ScenceDataset(object):

    def __init__(self):
        self.image_index = 0
        self.image_paths = []
        self.image_names = []
        self.image_labels = []
        self.image_store = []
        self.image_ids_group_by_class = dict()
        self.image_ids_group_by_store = dict()

    def add_image(self, img_path, label, store):
        if os.path.exists(img_path):
            img_name = os.path.basename(img_path)
            if label is None:
                label = int(os.path.dirname(img_path))

            self.image_paths.append(img_path)
            self.image_names.append(img_name)
            self.image_labels.append(label)
            self.image_store.append(store)
            if label not in self.image_ids_group_by_class:
                self.image_ids_group_by_class[label] = []
            self.image_ids_group_by_class[label].append(self.image_index)

            if store not in self.image_ids_group_by_store:
                self.image_ids_group_by_store[store] = []
            self.image_ids_group_by_store[store].append(self.image_index)
            self.image_index += 1
        else:
            print('Add not exists image {}'.format(img_path))

    def load_scene_db(self, db_patph):
        for store_name in os.listdir(db_patph):
            scene_root = os.path.join(db_patph, store_name)
            scene_dir = os.path.join(scene_root, 'annotaions')

            sameplace_dir = os.path.join(scene_dir, '01_sameplace')
            if os.path.exists(sameplace_dir):
                for img_name in os.listdir(sameplace_dir):
                    img_path = os.path.join(sameplace_dir, img_name)
                    if 'jpg' in img_name:
                        self.add_image(img_path, label=0, store=store_name)

            notsameplace_dir = os.path.join(scene_dir, '02_notsameplace')
            if os.path.exists(notsameplace_dir):
                for img_name in os.listdir(notsameplace_dir):
                    img_path = os.path.join(notsameplace_dir, img_name)
                    if 'jpg' in img_name:
                        self.add_image(img_path, label=1, store=store_name)
        print('loaded db {} instances'.format(len(self.image_paths)))

    def save_scene_db(self, out_db_path):
        for store_name, img_indexes in tqdm(self.image_ids_group_by_store.items()):
            if len(img_indexes) < 10:
                print('skipping store {} , number of image {} less than 10'.format(
                    store_name,len(img_indexes)))
                continue
            scene_root = os.path.join(out_db_path, store_name)
            scene_dir = os.path.join(scene_root, 'annotaions')
            sameplace_dir = os.path.join(scene_dir, '01_sameplace')
            notsameplace_dir = os.path.join(scene_dir, '02_notsameplace')

            mkdirs(sameplace_dir)
            mkdirs(notsameplace_dir)

            for index in img_indexes:
                ori_path = self.image_paths[index]
                ori_name = self.image_names[index]
                label = self.image_labels[index]

                if label == 0:
                    new_path = os.path.join(sameplace_dir, ori_name)
                elif label == 1:
                    new_path = os.path.join(notsameplace_dir, ori_name)
                copy(ori_path, new_path)


def mkdirs(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


if __name__ == "__main__":
    db = ScenceDataset()
    db.load_scene_db('/root/data/new_restaurant/')
    db.save_scene_db('/root/data/new_restaurant3/')
