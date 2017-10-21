from builtins import Exception

import numpy as np
import pandas as pd
import os, shutil
import h5py
import glob
import cv2
from subprocess import call
from keras.preprocessing import sequence
from sklearn.utils import shuffle


class MultimodalDataset(object):

    def __init__(self):
        self.lst_x = []
        self.lst_y = []
        self.activity_dict = {
            'act01': (0, 'walking'), 'act02': (1, 'walking upstairs'), 'act03': (2, 'walking downstairs'),
            'act04': (3, 'riding elevator up'), 'act05': (4, 'riding elevator down'), 'act06':(5, 'riding escalator up'),
            'act07': (6, 'riding escalator down'), 'act08': (7, 'sitting'), 'act09': (8, 'eating'), 'act10': (9, 'drinking'),
            'act11': (10, 'texting'), 'act12': (11, 'phone calls'), 'act13': (12, 'working on pc'), 'act14': (13, 'reading'),
            'act15': (14, 'writing sentences'), 'act16': (15, 'organizing files'), 'act17': (16, 'running'), 'act18': (17, 'push-ups'),
            'act19': (18, 'sit-ups'), 'act20': (19, 'cycling')

        }

        self.sensor_columns = ['accx', 'accy', 'accz',
                               'grax', 'gray', 'graz',
                               'gyrx', 'gyry', 'gyrz',
                               'lacx', 'lacy', 'lacz',
                               'magx', 'magy', 'magz',
                               'rotx', 'roty', 'rotz', 'rote']

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
    def flow_image_from_dir(root, max_frames_per_video=450, batch_size=10, group_size=1):
        x = []
        y = []

        activity_dict = {
            'act01': (0, 'walking'), 'act02': (1, 'walking upstairs'), 'act03': (2, 'walking downstairs'),
            'act04': (3, 'riding elevator up'), 'act05': (4, 'riding elevator down'),
            'act06': (5, 'riding escalator up'),
            'act07': (6, 'riding escalator down'), 'act08': (7, 'sitting'), 'act09': (8, 'eating'),
            'act10': (9, 'drinking'),
            'act11': (10, 'texting'), 'act12': (11, 'phone calls'), 'act13': (12, 'working on pc'),
            'act14': (13, 'reading'),
            'act15': (14, 'writing sentences'), 'act16': (15, 'organizing files'), 'act17': (16, 'running'),
            'act18': (17, 'push-ups'),
            'act19': (18, 'sit-ups'), 'act20': (19, 'cycling')

        }
        while True:
            files = sorted(glob.glob(os.path.join(root, '*', '*', '*.jpg')))
            all_grouped_files = list()
            if group_size > 1:
                cur_activity = ""
                cur_seq = ""

                for img_file in files:

                    img_file_split = img_file.split(os.path.sep)
                    cur_ix = int(img_file_split[-1].split(".")[0].split("_")[-1])

                    if cur_activity != img_file_split[-3] or cur_seq != img_file_split[-2]:
                        cur_activity = img_file_split[-3]
                        cur_seq = img_file_split[-2]
                        grouped_files = list()

                    if len(grouped_files) < group_size and cur_ix <= max_frames_per_video:

                        grouped_files.append(img_file)
                        cur_ix += 1

                    if len(grouped_files) == group_size:
                        all_grouped_files.append(grouped_files)
                        grouped_files = []

            files = all_grouped_files if len(all_grouped_files) > 0 else files
            np.random.shuffle(files)

            for img_ix, img in enumerate(files):
                cur_img_batch = []
                if img_ix < max_frames_per_video:
                    if type(img) is not list:
                        img = [img]

                    activity = img[0].split(os.path.sep)[-3]

                    for img_file in img:

                        cur_img = cv2.resize(cv2.imread(img_file), (224, 224)).astype('float')

                        cur_img /= 255.
                        cur_img -= 0.5
                        cur_img *= 2.

                        cur_img_batch.append(cur_img)

                    x.append(np.squeeze(cur_img_batch))

                    y.append(activity_dict[activity][0])

                    if len(x) == batch_size:
                        #print(img)
                        yield np.array(x), np.eye(20)[np.array(y).astype(int)]
                        x, y = ([], [])

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
            self.x_train, self.y_train = self.split_windows(group_size=group_size,
                                                            step_size=step_size, X=self.x_train, y=self.y_train)
            self.x_test, self.y_test = self.split_windows(group_size=group_size,
                                                          step_size=step_size, X=self.x_test, y=self.y_test)


        print("Train {}.{}".format(self.x_train.shape, self.y_train.shape))
        print("Test {}.{}".format(self.x_test.shape, self.y_test.shape))

    def _load_sensor_from_file(self, actvity, activities_files, selected_sensors, sensor_root):
        list_x = []
        list_y = []
        for file in activities_files:
            df = pd.read_csv(sensor_root+"/"+file, index_col=None, header=None)
            df.columns = self.sensor_columns
            df = df[selected_sensors] if len(selected_sensors) > 0 else df
            sensor = sequence.pad_sequences(df.values.T, maxlen=150, dtype='float32')
            list_x.append(sensor.T)
            list_y.append([self.activity_dict[actvity][0]])

        return np.array(list_x), self.one_hot(np.array(list_y))

    def split_train_test_sns(self, sensor_root, test_seq=['seq05', 'seq10']):
        train_path = os.path.join(sensor_root, 'train')
        test_path = os.path.join(sensor_root, 'test')

        if not os.path.exists(train_path):
            os.mkdir(train_path)
        if not os.path.exists(test_path):
            os.mkdir(test_path)

        activity_files = glob.glob(os.path.join(sensor_root, '*.csv'))

        for file in activity_files:
            seq = file.split(os.path.sep)[-1][5:10]
            if seq not in test_seq:
                shutil.move(file, train_path)
            else:
                shutil.move(file, test_path)

    def load_all_sensor_files(self,selected_sensors, sensor_root):
        list_x = []
        list_y = []
        activities_files = glob.glob(os.path.join(sensor_root, '*.csv'))
        for file in activities_files:
            activity = file.split(os.path.sep)[-1][0:5]
            df = pd.read_csv(file, index_col=None, header=None)
            df.columns = self.sensor_columns
            df = df[selected_sensors] if len(selected_sensors) > 0 else df
            sensor = sequence.pad_sequences(df.values.T, maxlen=150, dtype='float32')
            list_x.append(sensor.T)
            list_y.append([self.activity_dict[activity][0]])

        return np.array(list_x), np.array(list_y)

    def one_hot(self, y_):
        # Function to encode output labels from number indexes
        # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]

        y_ = y_.reshape(len(y_))
        n_values = np.max(y_) + 1
        return np.eye(n_values)[np.array(y_, dtype=np.int32)]  # Returns FLOATS

    def split_windows(self, group_size, step_size, X, y):
        number_groups = int(((X.shape[1]-group_size)/step_size)+1)
        split_xy = [(X[j, i:i + group_size], y[j])
                    for j in range(len(X)) for i in range(0, number_groups * step_size, step_size)]
        split_x = np.array([x[0] for x in split_xy])
        split_y = np.array([y[1] for y in split_xy])
        return split_x, split_y

    # Video Extraction functions

    def greedy_split(self, arr, chunk_size, step_size, axis=0):
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


    def generate_hdf5(self, chunk_size, step_size, image_root, sensor_root,
                                sensors=['accx', 'accy', 'accz'], output_file='multimodal.hdf5',
                                sensor_total_samples=150, image_total_samples=450):

        max_samples = max([sensor_total_samples, image_total_samples])
        min_samples = min([sensor_total_samples, image_total_samples])

        min_max_raio = max_samples / min_samples

        chunk_size_sensor = chunk_size if sensor_total_samples > image_total_samples else chunk_size / min_max_raio
        chunk_size_image = chunk_size if image_total_samples > sensor_total_samples else chunk_size / min_max_raio

        step_size_sensor = step_size if sensor_total_samples > image_total_samples else step_size / min_max_raio
        step_size_image = step_size if image_total_samples > sensor_total_samples else step_size / min_max_raio

        if not float(chunk_size_sensor).is_integer() or not float(chunk_size_image).is_integer():
            raise Exception("Both Chunk sizes should be a whole number")

        if not float(step_size_sensor).is_integer() or not float(step_size_image).is_integer():
            raise Exception("Both Step sizes should be a whole number")

        x_img = []
        y_img = []
        x_sensor = []
        y_sensor = []
        hf = h5py.File(output_file, "w")

        x_img_h5 = hf.create_dataset("x_img", shape=(0, 450,224, 224, 3), maxshape=(None, 450, 224, 224, 3))
        y_h5 = hf.create_dataset("y", shape=(0,), maxshape=(None,))
        x_sns_h5 = hf.create_dataset("x_sns", shape=(0, 150, 6), maxshape=(None,150, 6))
        global_ix = 1
        for path, subdirs, files in os.walk(image_root):
            if len(subdirs) > 0:
                print("Current Path: " + path)

            for ix, seq in enumerate(subdirs):
                cur_class = []
                cur_image = []

                files = glob.glob(path + "/" + seq + '/*.jpg')

                if len(files) > 0:
                    print("Creating images for: " + seq)

                for name in files:
                    if ix < image_total_samples:
                        cur_image = cv2.resize(cv2.imread(name), (224, 224))
                        cur_class.append(cur_image)

                while 0 < len(cur_class) < image_total_samples:
                    cur_class.append(cur_image)

                if len(cur_class) > 0:
                    act = path.split("/")[-1]
                    cur_class_arr = np.array(cur_class)

                    cur_class = self.greedy_split(cur_class_arr,
                                                  chunk_size=chunk_size_image, step_size=step_size_image)
                    csv_file = act + seq + '.csv'
                    sensor_x, sensor_y = self._load_sensor_from_file(actvity=act, activities_files=[csv_file],
                                                                     selected_sensors=sensors,
                                                                     sensor_root=sensor_root)

                    sensor_x, sensor_y = self.split_windows(int(chunk_size_sensor), int(step_size_sensor),
                                                                 sensor_x, sensor_y)

                    for cur_frame, cur_sensor_x, cur_sensor_y in zip(cur_class, sensor_x, sensor_y):
                        x_img.append(cur_frame)
                        y_img.append(self.activity_dict[path.split('/')[-1]][0])
                        x_sensor.append(cur_sensor_x)
                        y_sensor.append(self.activity_dict[path.split('/')[-1]][0])

                    if global_ix !=0 and global_ix % 10 == 0:
                        cur_ix = x_img_h5.shape[0]
                        x_img_h5.resize((cur_ix+10, 450, 224,224, 3))
                        x_sns_h5.resize((cur_ix+10, 150, 6))
                        y_h5.resize((cur_ix+10,))

                        x_img_h5[cur_ix:] = x_img
                        x_sns_h5[cur_ix:] = x_sensor
                        y_h5[cur_ix:] = y_sensor
                        x_img = []
                        x_sensor = []
                        y_sensor = []

                    global_ix += 1

                    print(x_img_h5.shape)
                    print(x_sns_h5.shape)
                    print(y_h5.shape)
        hf.close()
        # x, y = shuffle(x, y)
        if len(x_sensor) != len(x_img):
            raise Exception("Dataset sizes do not match")



            # if len(x) > 0:
            #     print(min([len(p) for p in x]))
            # x = []
            # y = []

    def extract_files(self, test_seqs=['seq09', 'seq10'], folders='../multimodal_dataset/video'):
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
                    video_parts = self.get_video_parts(video_path)

                    filename, class_label, seq, video_dir = video_parts

                    # Only extract if we haven't done it yet. Otherwise, just get
                    # the info.
                    if seq in test_seqs:
                        prefix = "test/"
                    else:
                        prefix = "train/"

                    cur_class_dir = video_dir + "images/" + prefix + class_label + '/'
                    if not bool(os.path.exists(cur_class_dir + '/' + seq)):
                        os.makedirs(cur_class_dir + '/' + seq)

                    src = video_dir + '/' + filename
                    dest = cur_class_dir + seq + '/' + class_label + '_' + seq + '_%04d.jpg'
                    call(["ffmpeg", "-i", src, dest])

                    # Now get how many frames it is.
                    # nb_frames = get_nb_frames_for_video(video_parts)

                    # data_file.append([train_or_test, classname, filename_no_ext, nb_frames])

                    # print("Generated %d frames for %s" % (nb_frames, filename_no_ext))

        # with open('data_file.csv', 'w') as fout:
        #     writer = csv.writer(fout)
        #     writer.writerows(data_file)

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
