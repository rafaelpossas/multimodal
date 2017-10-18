import glob
from subprocess import call
import os
import numpy as np
import pandas as pd
import cv2
import h5py
import shutil

activity_dict = ['walking', 'walking down/upstairs', 'riding elevator', 'riding escalator', 'working on pc', 'reading',
                 'writing', 'eating and drinking', 'browsing mobile phone', 'running', 'doing push ups', 'doing sit ups',
                 'cycling', 'washing dishes', 'watching tv', 'chopping food', 'cooking on stove', 'brushing teeth', 'lying down',
                 'talking with people']


def get_video_parts(video_path):
    """Given a full path to a video, return its parts."""
    parts = video_path.split('/')
    filename = parts[-1]
    dir = '/'.join(parts[0:-1]) + '/'
    return filename, dir



def assign_labels(folders=['vuzix/'], sensor_file_prefix='ACC_REC', labels_file_name='labels'):
    img_array = None
    label_array = None

    for folder in folders:
        class_folders = glob.glob(folder + '*')
        for global_ix, vid_class in enumerate(class_folders):

            all_img_files = glob.glob(vid_class + '/images/*.jpg')
            grouped_imag_files = glob.glob(vid_class + '/images/*/*.jpg')

            sensor_file = glob.glob(vid_class + '/'+sensor_file_prefix+'*.txt')
            labels_file = glob.glob(vid_class + '/'+labels_file_name+'*.txt')

            cur_labels = pd.read_csv(labels_file[0], delimiter=':', header=None)
            cur_labels = np.array(cur_labels.iloc[:, 1], dtype=pd.Series)

            sensors_array = pd.read_csv(sensor_file[0], delimiter='\t', header=None)
            sensors_array = np.array(sensors_array.iloc[:, 3], dtype=pd.Series)
            sensors_array = [(value,cur_labels[int(value)]) for value in sensors_array]

            for file in all_img_files:
                cur_index = int(int(file.split("_")[-1].split(".")[0]))
                try:
                    dest_dir = os.path.join(vid_class, "images", "_".join([str(sensors_array[cur_index][0]),
                                                                           sensors_array[cur_index][1]]))
                except IndexError:
                    os.remove(file)
                    file = None

                finally:

                    if not os.path.exists(dest_dir):
                        os.mkdir(dest_dir)

                    if file is not None:
                        shutil.move(file, dest_dir)

            if img_array is None and label_array is None:
                img_array = np.empty((len(class_folders), len(sensors_array)), dtype=np.ndarray)
                label_array = np.empty((len(class_folders), len(sensors_array)), dtype=np.ndarray)

            for ix, img in enumerate(grouped_imag_files):
                try:
                    cur_image = cv2.resize(cv2.imread(img), (224, 224))

                    img_array[global_ix][ix] = cur_image
                    label_array[global_ix][ix] = sensors_array[ix]
                except IndexError:
                    pass
    return img_array, label_array

def extract_files(folders=['vuzix/']):
    """After we have all of our videos split between train and test, and
    all nested within folders representing their classes, we need to
    make a data file that we can reference when training our RNN(s).
    This will let us keep track of image sequences and other parts
    of the training process.
    We'll first need to extract images from each of the videos. We'll
    need to record the following data in the file:
    [train|test], class, filename, nb frames
    Extracting can be done with ffmpeg:
    `ffmpeg -i video.mpg image-%04d.jpg`
    """
    data_file = []

    for folder in folders:
        class_folders = glob.glob(folder + '*')

        for vid_class in class_folders:
            class_files = glob.glob(vid_class + '/*.mp4')

            for video_path in class_files:
                # Get the parts of the file.
                video_parts = get_video_parts(video_path)

                filename, video_dir = video_parts

                # Only extract if we haven't done it yet. Otherwise, just get
                # the info.
                # if seq in test_seqs:
                #     prefix = "test/"
                # else:
                #     prefix = "train/"

                cur_class_dir = video_dir + "images/"
                if not bool(os.path.exists(cur_class_dir)):
                    os.makedirs(cur_class_dir)

                src = video_dir + '/' + filename
                dest = cur_class_dir + filename.split['.'][0] + '_%04d.jpg'
                call(["ffmpeg", "-i", src, dest])

                # Now get how many frames it is.
                # nb_frames = get_nb_frames_for_video(video_parts)

                # data_file.append([train_or_test, classname, filename_no_ext, nb_frames])

                # print("Generated %d frames for %s" % (nb_frames, filename_no_ext))

    # with open('data_file.csv', 'w') as fout:
    #     writer = csv.writer(fout)
    #     writer.writerows(data_file)

    print("Extracted and wrote %d video files." % (len(data_file)))

if __name__=='__main__':
    extract_files()
    assign_labels()