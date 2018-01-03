import os


from scipy.misc import imread


def get_image_list(data_dir, db):
    path_one = os.path.join(data_dir, 'Skin Image Data Set-1/skin_data/melanoma/')
    path_two = os.path.join(data_dir, 'Skin Image Data Set-2/skin_data/notmelanoma/')
    melanoma = [os.path.join(path_one, db, item.split('_')[0])
                for item in os.listdir(os.path.join(path_one, db)) if not item.endswith('db')]
    not_melanoma = [os.path.join(path_two, db, item.split('_')[0])
                    for item in os.listdir(os.path.join(path_two, db)) if not item.endswith('db')]
    return melanoma, not_melanoma


def load_one_database(data_dir, db):
    images = []
    labels = []
    melanoma, not_melanoma = get_image_list(data_dir, db)
    all_files = melanoma + not_melanoma
    for item in all_files:
        image_path = item + '_orig.jpg'
        label_path = item + '_contour.png'
        image = imread(image_path)
        label = imread(label_path)
        images.append(image)
        labels.append(label)
    return images, labels, all_files


class SkinData:
    def __init__(self, data_dir, db):
        self.db = db
        self.images, self.labels, self.listing = load_one_database(data_dir, db)

    def show_shapes(self):
        for i in range(len(self.images)):
            print('{:<7} shape{}'.format(self.listing[i].split('/')[-1], self.images[i].shape))

    def __repr__(self):
        myself = '\n'
        myself += 'database:  %s\n' % self.db
        myself += '#examples: %s\n' % len(self.images)
        return myself


if __name__ == '__main__':
    dermis = SkinData('/home/wenfeng/Desktop/Image-Segmentations/icpr-2018/data', 'dermis')
    print(dermis)
    dermis.show_shapes()
    dermquest = SkinData('/home/wenfeng/Desktop/Image-Segmentations/icpr-2018/data', 'dermquest')
    print(dermquest)
    dermquest.show_shapes()

