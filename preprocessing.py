import os
import sys


from scipy.misc import imread, imsave


def get_image_list(data_dir, db):
    path_one = os.path.join(data_dir, 'Skin Image Data Set-1/skin_data/melanoma/')
    path_two = os.path.join(data_dir, 'Skin Image Data Set-2/skin_data/notmelanoma/')
    melanoma = [os.path.join(path_one, db, item.split('_')[0])
                for item in os.listdir(os.path.join(path_one, db)) if not item.endswith('db')]
    not_melanoma = [os.path.join(path_two, db, item.split('_')[0])
                    for item in os.listdir(os.path.join(path_two, db)) if not item.endswith('db')]
    return melanoma, not_melanoma


def change255to1(data_dir):
    one, two = get_image_list(data_dir, 'dermis')
    three, four = get_image_list(data_dir, 'dermquest')
    for item in one + two + three + four:
        label_path = item + '_contour.png'
        label = imread(label_path)
        label[label == 255] = 1
        imsave(label_path, label)


if __name__ == '__main__':
    # print(sys.argv)
    change255to1(sys.argv[1])
