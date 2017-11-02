import glob
from subprocess import call
import os
import numpy as np
import pandas as pd
import cv2
import h5py
import shutil
from MultimodalDataset import MultimodalDataset
from globals import activity_dict_vuzix
from scipy.stats import mode

class VuzixDataset:


    def get_video_parts(self, video_path):
        """Given a full path to a video, return its parts."""
        parts = video_path.split('/')
        filename = parts[-1]
        dir = parts[1]
        return filename, dir

    @staticmethod
    def flow_images_from_dir(root="vuzix/test/", max_frames_per_video=4500, batch_size=450):
        all_directories = glob.glob(os.path.join(root, '*'))
        np.random.shuffle(all_directories)
        x = []
        y = []
        while True:
            for dir in all_directories:
                files = glob.glob(os.path.join(dir, "images", "*", '*.jpg'))
                labels = np.load(os.path.join(dir, "acc_gyr.npy"))[:, -1]
                for img_ix, img in enumerate(sorted(files)):
                    if img_ix < max_frames_per_video:

                        cur_img = cv2.resize(cv2.imread(img), (224, 224)).astype('float')

                        cur_img /= 255.
                        cur_img -= 0.5
                        cur_img *= 2.

                        x.append(cur_img)

                        y.append(labels[img_ix])

                        if len(x) == batch_size:
                            # print(img)
                            yield np.array(x), np.eye(20)[np.array(y).astype(int)]
                            x, y = ([], [])
    @staticmethod
    def get_total_size(image_root):
        recordings = glob.glob(os.path.join(image_root, '*', '*.npy'))
        recordings = len(recordings) / 2
        return recordings

    def assign_labels(self, folders=['vuzix/images/'],
                      sensor_files=['ACC', 'GYR'], labels_file_name='labels'):

        for folder in sorted(folders):
            class_folders = glob.glob(folder + '*')
            for global_ix, vid_class in enumerate(sorted(class_folders)):
                label_array = None
                sensors_array = None
                print(vid_class)
                all_img_files = glob.glob(os.path.join(vid_class, '*.jpg'))

                labels_file = glob.glob(vid_class + '/'+labels_file_name+'*.txt')
                cur_labels = pd.read_csv(labels_file[0], delimiter=':', header=None)
                cur_labels = np.array(cur_labels.iloc[:, 1], dtype=pd.Series)

                for sns in sensor_files:
                    sensor_file = glob.glob(os.path.join(vid_class, sns+'*'))
                    df = np.loadtxt(sensor_file[0], delimiter="\t")
                    if sensors_array is None:
                        sensors_array = df[:, :3]
                        label_array = df[:, -1]
                    else:
                        sensors_array = np.column_stack((sensors_array, df[:, :3]))
                        labels = df[:, -1]
                        if not np.array_equal(label_array, labels):
                            comp = label_array == labels
                            if len(np.where(comp is False)) > 10:
                                raise ValueError("Error on recording {} Label arrays between sensors are different"
                                                 .format(vid_class))

                label_array = label_array.astype("int")
                real_labels_array = [activity_dict_vuzix().index(cur_labels[int(value)].rstrip()) for value in label_array]

                np.save(os.path.join(vid_class, "acc_gyr_x"), sensors_array)
                np.save(os.path.join(vid_class, "acc_gyr_y"), real_labels_array)

                for file in sorted(all_img_files):
                    cur_index = int(int(file.split("_")[-1].split(".")[0]))
                    dest_dir = None
                    try:
                        dest_dir = os.path.join(vid_class, "_".join([str(label_array[cur_index-1]), activity_dict_vuzix()[real_labels_array[cur_index-1]]
                                                                              .replace("/", "_")]))
                    except IndexError:
                        os.remove(file)
                        file = None

                    finally:

                        if not os.path.exists(dest_dir):
                            os.mkdir(dest_dir)

                        if file is not None:
                            shutil.move(file, dest_dir)

    def split_test_train(self, root="vuzix/images/", test_percentage=0.20,
                         imgs_per_video=4500, group_size=15):

        recordings = glob.glob(os.path.join(root, "*"))
        samples_per_video = (imgs_per_video/group_size) * test_percentage
        for rec in sorted(recordings):
            print(rec)
            all_img = glob.glob(os.path.join(rec, '*', '*.jpg'))
            use_full_sequence = np.random.choice([True, False], p=[0.10, 0.90])

            sns_arr_x = np.load(os.path.join(rec, "acc_gyr_x.npy"))
            sns_arr_y = np.load(os.path.join(rec, "acc_gyr_y.npy"))

            sns_arr_x, sns_arr_y_onehot = VuzixDataset.group_sensors(sns_arr_x, sns_arr_y)
            img_x, img_y_onehot = VuzixDataset.group_sensors(sorted(all_img), sns_arr_y)

            if not use_full_sequence:
                rand_ix = np.random.randint(0, int((imgs_per_video/group_size)) - samples_per_video)
                range_ix = range(rand_ix, int(rand_ix + samples_per_video))
            else:
                range_ix = range(0, int(imgs_per_video/group_size))

            train_img_x = [file for ix, file in enumerate(img_x) if (ix not in list(range_ix) and ix < imgs_per_video/group_size)]
            test_img_x = [file for ix, file in enumerate(img_x) if (ix in list(range_ix) and ix < imgs_per_video/group_size)]

            train_sns_x = [file for ix, file in enumerate(sns_arr_x) if (ix not in list(range_ix) and ix < imgs_per_video/group_size)]
            train_sns_y = [y for ix, y in enumerate(sns_arr_y_onehot) if (ix not in list(range_ix) and ix < imgs_per_video / group_size)]

            test_sns_x = [file for ix, file in enumerate(sns_arr_x) if (ix in list(range_ix) and ix < imgs_per_video/group_size)]
            test_sns_y = [y for ix, y in enumerate(sns_arr_y_onehot) if (ix in list(range_ix) and ix < imgs_per_video / group_size)]

            dest = os.path.join(root, "splits", "train", rec.split(os.path.sep)[-1])

            for ix, (img_x, sns_x, label) in enumerate(zip(train_img_x, train_sns_x, train_sns_y)):
                for img_file in img_x:
                    dest_img = os.path.join(dest, *img_file.split(os.path.sep)[-2:-1])

                    if not os.path.exists(dest_img):
                        os.makedirs(dest_img, exist_ok=True)

                    shutil.copy(img_file, dest_img)

            if os.path.exists(dest):
                dest_sns_x = os.path.join(dest, "sns_x.npy")
                dest_sns_y = os.path.join(dest, "sns_y.npy")
                np.save(dest_sns_x, train_sns_x)
                np.save(dest_sns_y, train_sns_y)

            dest = os.path.join(root, "splits", "test", rec.split(os.path.sep)[-1])

            for ix, (img_x, sns_x, label) in enumerate(zip(test_img_x, test_sns_x, test_sns_y)):
                for img_file in img_x:
                    dest_img = os.path.join(dest, *img_file.split(os.path.sep)[-2:-1])

                    if not os.path.exists(dest_img):
                        os.makedirs(dest_img, exist_ok=True)

                    shutil.copy(img_file, dest_img)

            if os.path.exists(dest):
                dest_sns_x = os.path.join(dest, "sns_x.npy")
                dest_sns_y = os.path.join(dest, "sns_y.npy")
                np.save(dest_sns_x, test_sns_x)
                np.save(dest_sns_y, test_sns_y)

    @staticmethod
    def get_all_files(root, group_size, max_samples=150):

        all_grouped_img_x = []
        all_grouped_sns_x = []
        all_grouped_sns_y = []

        if group_size > 1:
            recordings = sorted(glob.glob(os.path.join(root, '*')))
            for rec in recordings:
                files = sorted(glob.glob(os.path.join(rec, '*', '*.jpg')))
                sns_x = np.load(os.path.join(rec, "sns_x.npy"))
                sns_y = np.load(os.path.join(rec, "sns_y.npy"))
                grouped_files = []
                counter = 0
                for img_file in files:
                    img_file_split = img_file.split(os.path.sep)
                    cur_ix = int(img_file_split[-1].split(".")[0].split("_")[-1])

                    if len(grouped_files) < group_size and cur_ix <= max_samples:
                        grouped_files.append(img_file)

                    if len(grouped_files) == group_size:
                        all_grouped_img_x.append(grouped_files)
                        all_grouped_sns_x.append(sns_x[counter])
                        all_grouped_sns_y.append(sns_y[counter])
                        counter += 1
                        grouped_files = []
        else:
            files = sorted(glob.glob(os.path.join(root, '*', '*', '*.jpg')))
            all_grouped_img_x = np.array(all_grouped_img_x) if len(all_grouped_img_x) > 0 else np.array(files)

        return np.array(all_grouped_img_x), np.array(all_grouped_sns_x), np.array(all_grouped_sns_y)

    @staticmethod
    def flow_from_dir(root="vuzix/images/splits/train", max_frames_per_video=4500, batch_size=10, group_size=1,
                      resize_shape=(448, 256),
                      img_shape=(224, 224), type="img", shuffle_arrays=True):
        img_x = []
        sns_x = []
        y = []

        while True:
            all_grouped_img, all_grouped_sns, labels = VuzixDataset.get_all_files(root, group_size, max_frames_per_video)
            if len(all_grouped_img) == 0 and len(all_grouped_sns) == 0:
                raise Exception("There are no data in the arrays for the generator")
            if shuffle_arrays:
                arr_ix = np.arange(len(all_grouped_img))
                np.random.shuffle(arr_ix)
                all_grouped_img = all_grouped_img[arr_ix]

                if len(all_grouped_sns) > 0:
                    all_grouped_sns = all_grouped_sns[arr_ix]
                    labels = labels[arr_ix]

            for img_ix, img in enumerate(all_grouped_img):
                cur_img_batch = []
                sns = None

                if len(all_grouped_sns) > 0:
                    sns = all_grouped_sns[img_ix]

                if not isinstance(img, np.ndarray):
                    img = [img]

                if type != "sns":
                    for img_file in img:
                        cur_img = MultimodalDataset.get_img_from_file(img_file, resize_shape, img_shape)
                        cur_img_batch.append(cur_img)

                    img_x.append(np.squeeze(cur_img_batch))

                if sns is not None:
                    sns_x.append(sns)
                if len(labels) > 0:
                    y.append(labels[img_ix])
                else:
                    activity = img[0].split(os.path.sep)[-2].split("_")[1]
                    y.append(activity_dict_vuzix().index(activity))

                if len(img_x) == batch_size or len(sns_x) == batch_size:
                    # print(img)

                    if type == "img":
                        yield np.array(img_x), np.eye(20)[np.array(y).astype(int)]
                        #print("\n{} - Returning batch size of {}".format(img_ix, batch_size))

                    if type == "sns":
                        if len(sns_x) < batch_size:
                            print("Empty Sensor Bath")
                        yield np.array(sns_x), np.array(y)
                        #print("\n{} - Returning batch size of {}".format(img_ix, batch_size))

                    img_x, sns_x, y = ([], [], [])

    def extract_files(self, folders=['vuzix/']):
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

            for vid_class in sorted(class_folders):
                class_files = glob.glob(vid_class + '/*.mp4')

                for video_path in class_files:
                    # Get the parts of the file.
                    video_parts = self.get_video_parts(video_path)

                    filename, video_dir = video_parts

                    # Only extract if we haven't done it yet. Otherwise, just get
                    # the info.
                    # if seq in test_seqs:
                    #     prefix = "test/"
                    # else:
                    #     prefix = "train/"
                    # dataset = np.random.choice(['train', 'test'], p=[0.8, 0.2])
                    #
                    # if dataset == "test":
                    #     prefix = "test"
                    # else:
                    #     prefix = "train"

                    src = os.path.join(folder, video_dir, filename)
                    dest = os.path.join(folder, "images", video_dir)

                    if not bool(os.path.exists(dest)):
                        os.makedirs(dest)

                    dest = os.path.join(dest, filename.split('.')[0] + '_%04d.jpg')

                    call(["ffmpeg", "-i", src, dest])

                    txt_files = glob.glob(os.path.join(folder, video_dir, '*.txt'))

                    for file in txt_files:
                        shutil.copy(file, os.path.join(folder, "images", video_dir))

                    # Now get how many frames it is.
                    # nb_frames = get_nb_frames_for_video(video_parts)

                    # data_file.append([train_or_test, classname, filename_no_ext, nb_frames])

                    # print("Generated %d frames for %s" % (nb_frames, filename_no_ext))

        # with open('data_file.csv', 'w') as fout:
        #     writer = csv.writer(fout)
        #     writer.writerows(data_file)

        print("Extracted and wrote %d video files." % (len(data_file)))
    @staticmethod
    def get_all_sensor_files(root):
        all_recordings = glob.glob(os.path.join(root, 'REC*'))

        all_sns_acc_x = []
        all_sns_gyr_x = []
        all_sns_acc_y = []
        all_sns_gyr_y = []

        for recording in sorted(all_recordings):
            recording_number = recording.split("_")[-1]
            acc_file_path = os.path.join(recording, 'ACC_REC_'+recording_number+".txt")
            gyr_file_path = os.path.join(recording, 'GYR_REC_'+recording_number+".txt")
            labels = pd.read_csv(os.path.join(recording, 'labels.txt'), header=None, delimiter=":")
            cur_labels = np.array(labels.iloc[:, 1], dtype=pd.Series)

            sns_acc = np.loadtxt(acc_file_path)
            sns_gyr = np.loadtxt(gyr_file_path)
            x_acc = sns_acc[:, :3]
            y_acc = [activity_dict_vuzix().index(cur_labels[int(label_ix)].rstrip()) for label_ix in sns_acc[:, 3:]]
            x_gyr = sns_gyr[:, :3]
            y_gyr = [activity_dict_vuzix().index(cur_labels[int(label_ix)].rstrip()) for label_ix in sns_gyr[:, 3:]]
            all_sns_acc_x.append(x_acc)
            all_sns_acc_y.append(y_acc)
            all_sns_gyr_x.append(x_gyr)
            all_sns_gyr_y.append(y_gyr)

        all_sns_acc_x = np.array(all_sns_acc_x).reshape((-1, 3))
        all_sns_acc_y = np.reshape(all_sns_acc_y, (-1, 1))

        all_sns_gyr_x = np.array(all_sns_gyr_x).reshape((-1, 3))
        all_sns_gyr_y = np.reshape(all_sns_gyr_y, (-1, 1))

        return all_sns_acc_x, all_sns_acc_y, all_sns_gyr_x, all_sns_gyr_y

    @staticmethod
    def group_sensors(sns_x, sns_y):

        x_group = []
        y_group = []
        all_x = []
        all_y = []

        for x, y in zip(sns_x, sns_y):
            x_group.append(x)
            y_group.append(y)
            if len(x_group) == 15:
                all_x.append(x_group)
                all_y.append(np.squeeze(mode(y_group))[0])
                x_group = []
                y_group = []

        all_x = np.array(all_x)
        all_y = np.eye(20)[all_y]

        return all_x, all_y

if __name__=="__main__":
    dataset = VuzixDataset()
    #dataset.extract_files()
    #dataset.assign_labels()
    dataset.split_test_train()
    # generator = VuzixDataset.flow_from_dir(type="sns", group_size=15)
    # while True:
    #     next(generator)

