from builtins import Exception, staticmethod

import numpy as np
import pandas as pd
import os, shutil
import glob
import cv2
from subprocess import call
from keras.preprocessing import sequence
from sklearn.utils import shuffle
from globals import activity_dict, sensor_columns
import argparse


class MultimodalDataset(object):

    def __init__(self):
        self.lst_x = []
        self.lst_y = []

    def get_sensor_filepaths(self, directory):
        """
        This function will generate the file names in a directory
        tree by walking the tree either top-down or bottom-up. For each
        directory in the tree rooted at directory top (including top itself),
        it yields a 3-tuple (dirpath, dirnames, filenames).
        """
        file_paths = []  # List which will store all of the full filepaths.
        activities_files = {}
        # Walk the tree.
        for root, directories, files in os.walk(directory):
            for filename in files:
                # Join the two strings in order to form the full filepath.
                #filepath = os.path.join(root, filename)
                file_paths.append(filename)  # Add it to the list.
                activity = filename[0:5]
                if activity != '.DS_S':
                    if activity in activities_files.keys():
                        activities_files[activity].append(filename)
                    else:
                        activities_files[activity] = [filename]


        return activities_files  # Self-explanatory.

    @staticmethod
    def get_all_files(root, group_size, max_samples=150):
        files = sorted(glob.glob(os.path.join(root, '*', '*', '*.jpg')))
        all_grouped_img = []
        all_grouped_sns = []
        cur_activity = ""
        cur_seq = ""
        sns_file = None
        counter = 0
        if group_size > 1:
            for img_file in files:
                img_file_split = img_file.split(os.path.sep)
                cur_ix = int(img_file_split[-1].split(".")[0].split("_")[-1])

                if cur_activity != img_file_split[-3] or cur_seq != img_file_split[-2]:
                    cur_activity = img_file_split[-3]
                    cur_seq = img_file_split[-2]
                    grouped_files = list()
                    grouped_sns = list()
                    counter = 0
                    sns_file = None

                if sns_file is None:
                    sns_file = np.load(os.path.join(*img_file_split[0:-1], "sns.npy"))

                if len(grouped_files) < group_size and cur_ix <= max_samples:
                    grouped_files.append(img_file)

                    if counter < len(sns_file):
                        grouped_sns.append(sns_file[counter])
                        counter += 1

                if len(grouped_files) == group_size:
                    all_grouped_img.append(grouped_files)
                    all_grouped_sns.append(grouped_sns)
                    grouped_files = []
                    grouped_sns = []

        all_grouped_img = np.array(all_grouped_img) if len(all_grouped_img) > 0 else np.array(files)

        return all_grouped_img, all_grouped_sns

    @staticmethod
    def get_img_from_file(img_file, resize_shape=(448,256), img_shape=(224,224)):
        cur_img = cv2.resize(cv2.imread(img_file), resize_shape,
                             interpolation=cv2.INTER_AREA).astype('float')

        crop_width_start = np.random.randint(0, 224)
        crop_heigth_start = np.random.randint(0, 32)

        cur_img = cur_img[crop_heigth_start:img_shape[1] + crop_heigth_start,
                  crop_width_start:crop_width_start + img_shape[0], :]

        cur_img /= 255.
        # cur_img -= 0.5
        # cur_img *= 2.
        return cur_img

    @staticmethod
    def flow_from_dir(root, max_frames_per_video=450, batch_size=10, group_size=1, resize_shape=(448, 256),
                            img_shape=(224,224), type="img", shuffle_arrays=True):
        img_x = []
        sns_x = []
        y = []

        while True:
            all_grouped_img, all_grouped_sns = MultimodalDataset.get_all_files(root, group_size, max_frames_per_video)
            #print("Going through the entire batch on {}".format(root))
            if shuffle_arrays:
                arr_ix = np.arange(len(all_grouped_img))
                np.random.shuffle(arr_ix)
                all_grouped_img = all_grouped_img[arr_ix]

                if len(all_grouped_sns) > 0:
                    all_grouped_sns = np.array(all_grouped_sns)[arr_ix]

            for img_ix, img in enumerate(all_grouped_img):
                cur_img_batch = []
                sns = None
                if len(all_grouped_sns) > 0:
                    sns = all_grouped_sns[img_ix]

                if not isinstance(img, np.ndarray):
                    img = [img]

                activity = img[0].split(os.path.sep)[-3]

                if type != "sns":
                    for img_file in img:
                        cur_img = MultimodalDataset.get_img_from_file(img_file, resize_shape, img_shape)
                        cur_img_batch.append(cur_img)

                    img_x.append(np.squeeze(cur_img_batch))

                if sns is not None:
                    sns_x.append(sns)

                y.append(activity_dict()[activity][0])

                if len(img_x) == batch_size or len(sns_x) == batch_size:
                    #print(img)

                    if type == "img":
                        yield np.array(img_x), np.eye(20)[np.array(y).astype(int)]
                        #print("\n{} - Returning batch size of {}".format(img_ix, batch_size))

                    if type == "sns":
                        yield np.array(sns_x), np.eye(20)[np.array(y).astype(int)]
                        #print("\n{} - Returning batch size of {}".format(img_ix, batch_size))
                    img_x,sns_x, y = ([],[], [])

    def load_sensor_dataset(self, train_size=0.8, split_train=True,
                  group_size=50, step_size=0, selected_sensors=[], root_dir=None):

        act = self.get_sensor_filepaths(root_dir)
        self.lst_x, self.lst_y = self._load_all_files(act, selected_sensors)
        self.lst_x, self.lst_y = shuffle(self.lst_x, self.lst_y, random_state=0)

        train_size = int(len(self.lst_x)*train_size)

        self.x_train, self.x_test = self.lst_x[0:train_size, :, :], \
                                    self.lst_x[train_size:, :, :]

        self.y_train, self.y_test = self.lst_y[0:train_size, :], \
                                    self.lst_y[train_size:, :]
        if split_train:
            self.x_train, self.y_train = MultimodalDataset.split_windows(group_size=group_size,
                                                            step_size=step_size, X=self.x_train, y=self.y_train)
            self.x_test, self.y_test = MultimodalDataset.split_windows(group_size=group_size,
                                                          step_size=step_size, X=self.x_test, y=self.y_test)


        print("Train {}.{}".format(self.x_train.shape, self.y_train.shape))
        print("Test {}.{}".format(self.x_test.shape, self.y_test.shape))

    @staticmethod
    def load_sensor_from_file(file, selected_sensors):
        list_x = []

        df = pd.read_csv(file, index_col=None, header=None)
        df.columns = sensor_columns()
        df = df[selected_sensors] if len(selected_sensors) > 0 else df
        sensor = sequence.pad_sequences(df.values.T, maxlen=150, dtype='float32')
        list_x.append(sensor.T)

        return np.array(list_x)

    def split_test_train(self, img_root="multimodal_dataset/video/images/",
                         sns_root="multimodal_dataset/sensor/",
                         test_percentage=0.20, imgs_per_video=150, fps=10,
                         sensors=['accx', 'accy', 'accz', 'gyrx', 'gyry', 'gyrz']):

        activities = glob.glob(os.path.join(img_root, "*"))
        samples_per_video = imgs_per_video * test_percentage
        for act in sorted(activities):
            sequences = glob.glob(os.path.join(act, "*"))
            for seq in sorted(sequences):
                all_img = glob.glob(os.path.join(seq, '*.jpg'))
                use_full_sequence = np.random.choice([True, False], p=[0.1, 0.9])
                activity = seq.split(os.path.sep)[-2]
                sequence = seq.split(os.path.sep)[-1]

                sns_file = os.path.join(sns_root, activity + sequence + ".csv")
                sns_arr = MultimodalDataset.load_sensor_from_file(sns_file, sensors)

                if not use_full_sequence:
                    rand_ix = np.random.randint(0, int(imgs_per_video - samples_per_video) / fps)
                    rand_ix = rand_ix * fps
                    range_ix = range(rand_ix, int(rand_ix + samples_per_video))
                else:
                    range_ix = range(0, 150)

                train_img = sorted([file for ix, file in enumerate(sorted(all_img)) if (ix not in list(range_ix) and ix < 150)])
                test_img = sorted([file for ix, file in enumerate(sorted(all_img)) if (ix in list(range_ix) and ix < 150)])

                train_sns_x = [file for ix, file in enumerate(sns_arr[0][0]) if (ix not in list(range_ix) and ix < 150)]
                test_sns_x = [file for ix, file in enumerate(sns_arr[0][0]) if (ix in list(range_ix) and ix < 150)]

                dest = None
                for train_file in train_img:
                    dest = os.path.join(img_root, "splits", "train", *train_file.split(os.path.sep)[-3:-1])

                    if not os.path.exists(dest):
                        os.makedirs(dest, exist_ok=True)

                    shutil.copy(train_file, dest)

                if dest is not None and len(train_img) > 0:
                    np.save(os.path.join(dest, "sns"), train_sns_x)

                for test_file in test_img:
                    dest = os.path.join(img_root, "splits", "test", *test_file.split(os.path.sep)[-3:-1])

                    if not os.path.exists(dest):
                        os.makedirs(dest, exist_ok=True)

                    shutil.copy(test_file, dest)

                if dest is not None and len(test_img) > 0:
                    np.save(os.path.join(dest, "sns"), test_sns_x)

                if len(test_sns_x) + len(train_sns_x) < 150:
                    print("Warning, less than 150 sensor inputs were saved")

    def load_all_sensor_files(self,selected_sensors, sensor_root):
        list_x = []
        list_y = []
        activities_files = glob.glob(os.path.join(sensor_root, '*.csv'))
        for file in activities_files:
            activity = file.split(os.path.sep)[-1][0:5]
            df = pd.read_csv(file, index_col=None, header=None)
            df.columns = sensor_columns()
            df = df[selected_sensors] if len(selected_sensors) > 0 else df
            sensor = sequence.pad_sequences(df.values.T, maxlen=150, dtype='float32')
            list_x.append(sensor.T)
            list_y.append([activity_dict()[activity][0]])

        return np.array(list_x), np.array(list_y)

    @staticmethod
    def split_windows(group_size, step_size, X, y, row_idx=1):
        number_groups = int(((X.shape[row_idx]-group_size)/step_size)+1)
        split_xy = [(X[j, i:i + group_size], y[j])
                    for j in range(len(X)) for i in range(0, number_groups * step_size, step_size)]
        split_x = np.array([x[0] for x in split_xy])
        split_y = np.array([y[1] for y in split_xy])
        return split_x, split_y

    # Video Extraction functions
    @staticmethod
    def greedy_split(arr, chunk_size, step_size, axis=0):
        """Greedily splits an array into n blocks.

        Splits array arr along axis into n blocks such that:
            - blocks 1 through n-1 are all the same size
            - the sum of all block sizes is equal to arr.shape[axis]
            - the last block is nonempty, and not bigger than the other blocks

        Intuitively, this "greedily" splits the array along the axis by making
        the first blocks as big as possible, then putting the leftovers in the
        last block.
        """
        length = arr.shape[axis]

        # the indices at which the splits will occur
        ix = np.arange(0, length, step_size).astype(int)

        return [arr[i:i + int(chunk_size), :] for i in ix]

    @staticmethod
    def get_activities_by_index(act_indexes):
        act_str_arr = []

        for act_ix in act_indexes:
            if act_ix < 10:
                act_str = "act0" + str(act_ix)
            else:
                act_str = "act" + str(act_ix)
            act_str_arr.append(act_str)

        return act_str_arr

    @staticmethod
    def get_total_size(image_root):
        activities = glob.glob(os.path.join(image_root, '*'))
        counter = 0
        for act in activities:
            seqs = glob.glob(os.path.join(act, '*', '*.jpg'))
            counter += len(seqs)
        return counter

    def extract_files(self, fps=10, video_root="multimodal_dataset/video"):
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
        folders = [video_root]
        for folder in folders:
            class_folders = glob.glob(folder + '*')

            for vid_class in class_folders:
                class_files = glob.glob(vid_class + '/*.mp4')
                np.random.shuffle(class_files)
                for video_path in class_files:
                    # Get the parts of the file.
                    video_parts = self.get_video_parts(video_path)

                    filename, class_label, seq, video_dir = video_parts

                    # Only extract if we haven't done it yet. Otherwise, just get
                    # the info.
                    # dataset = np.random.choice(['train', 'test'], p=[0.85, 0.15])
                    # if dataset == "test":
                    #     prefix = "test/"
                    # else:
                    #     prefix = "train/"

                    cur_class_dir = video_dir + "images/" + class_label + '/'
                    if not bool(os.path.exists(cur_class_dir + '/' + seq)):
                        os.makedirs(cur_class_dir + '/' + seq)

                    src = video_dir + filename
                    dest = cur_class_dir + seq + '/' + class_label + '_' + seq + '_%04d.jpg'
                    call("ffmpeg -i " + src + " -vf fps=" + str(fps) + " " + dest, shell=True)

        print("Extracted and wrote %d video files." % (len(data_file)))

    def get_nb_frames_for_video(self, video_parts):
        """Given video parts of an (assumed) already extracted video, return
        the number of frames that were extracted."""
        train_or_test, classname, filename_no_ext, _ = video_parts
        generated_files = glob.glob(train_or_test + '/' + classname + '/' +
                                    filename_no_ext + '*.jpg')
        return len(generated_files)

    def get_video_parts(self, video_path):
        """Given a full path to a video, return its parts."""
        parts = video_path.split('/')
        filename = parts[-1]
        class_label = filename[0:5]
        seq = filename[5:10]
        dir = '/'.join(parts[0:-1]) + '/'
        return filename, class_label, seq, dir

    @staticmethod
    def unison_shuffled_copies(a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]

if __name__=="__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--img_root", default="multimodal_dataset/video/images/", type=str)
    a.add_argument("--sns_root", default="multimodal_dataset/sensor/", type=str)
    a.add_argument("--video_root", default="multimodal_dataset/video", type=str)
    a.add_argument("--extract_videos", action='store_true')
    a.add_argument("--split_test_train", action="store_true")
    a.add_argument("--fps", default=10, type=int)

    args = a.parse_args()

    multimodal_dataset = MultimodalDataset()

    if args.extract_videos:
        multimodal_dataset.extract_files(fps=args.fps, video_root=args.video_root)
    if args.split_test_train:
        multimodal_dataset.split_test_train(img_root=args.img_root, sns_root=args.sns_root)
