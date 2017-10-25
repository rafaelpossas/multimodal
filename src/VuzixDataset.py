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
                            if len(np.where(comp == False)) > 10:
                                raise ValueError("Error on recording {} Label arrays between sensors are different"
                                                 .format(vid_class))

                label_array = label_array.astype("int")
                real_labels_array = [activity_dict_vuzix().index(cur_labels[int(value)].rstrip()) for value in label_array]
                sensors_array = np.column_stack((sensors_array, real_labels_array))

                np.save(os.path.join(vid_class, "acc_gyr"), sensors_array)

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

    def split_test_train(self, root="vuzix/images/",
                         test_percentage=0.20, imgs_per_video=4500, fps=15):
        recordings = glob.glob(os.path.join(root, "*"))
        samples_per_video = imgs_per_video * test_percentage
        for rec in sorted(recordings):

            all_img = glob.glob(os.path.join(rec, '*', '*.jpg'))
            use_full_sequence = np.random.choice([True, False], p=[0.1, 0.9])

            sns_arr = np.load(os.path.join(rec, "acc_gyr.npy"))

            if not use_full_sequence:
                rand_ix = np.random.randint(0, int(imgs_per_video - samples_per_video) / fps)
                rand_ix = rand_ix * fps
                range_ix = range(rand_ix, int(rand_ix + samples_per_video))
            else:
                range_ix = range(0, 150)

            train_img = sorted([file for ix, file in enumerate(sorted(all_img)) if (ix not in list(range_ix) and ix < imgs_per_video)])
            test_img = sorted([file for ix, file in enumerate(sorted(all_img)) if (ix in list(range_ix) and ix < imgs_per_video)])

            train_sns_x = [file for ix, file in enumerate(sns_arr) if (ix not in list(range_ix) and ix < imgs_per_video)]
            test_sns_x = [file for ix, file in enumerate(sns_arr) if (ix in list(range_ix) and ix < imgs_per_video)]

            dest = None
            for train_file in train_img:
                dest = os.path.join(root, "splits", "train", *train_file.split(os.path.sep)[-3:-1])

                if not os.path.exists(dest):
                    os.makedirs(dest, exist_ok=True)

                shutil.copy(train_file, dest)

            if dest is not None and len(train_img) > 0:
                np.save(os.path.join(*dest.split(os.path.sep)[:-1], "sns"), train_sns_x)

            for test_file in test_img:
                dest = os.path.join(root, "splits", "test", *test_file.split(os.path.sep)[-3:-1])

                if not os.path.exists(dest):
                    os.makedirs(dest, exist_ok=True)

                shutil.copy(test_file, dest)

            if dest is not None and len(test_img) > 0:
                np.save(os.path.join(*dest.split(os.path.sep)[:-1], "sns"), test_sns_x)

            if len(test_sns_x) + len(train_sns_x) < imgs_per_video:
                print("Warning, less than 4500 sensor inputs were saved")

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
    def flow_from_dir(root="vuzix/images/splits/train", max_frames_per_video=4500, batch_size=10, group_size=1, resize_shape=(448, 256),
                            img_shape=(224,224), type="img", shuffle_arrays=True):
        img_x = []
        sns_x = []
        y = []

        while True:
            all_grouped_img, all_grouped_sns = VuzixDataset.get_all_files(root, group_size, max_frames_per_video)

            if shuffle_arrays:
                arr_ix = np.arange(len(all_grouped_img))
                np.random.shuffle(arr_ix)
                all_grouped_img = all_grouped_img[arr_ix]

                if len(all_grouped_sns) > 0:
                    all_grouped_sns = np.array(all_grouped_sns)[arr_ix]

            for img_ix, img in enumerate(all_grouped_img):
                cur_img_batch = []
                if img_ix < max_frames_per_video:
                    sns = None

                    if len(all_grouped_sns) > 0:
                        sns = all_grouped_sns[img_ix]

                    if not isinstance(img, np.ndarray):
                        img = [img]

                    activity = img[0].split(os.path.sep)[-2].split("_")[1]

                    if type != "sns":
                        for img_file in img:
                            cur_img = MultimodalDataset.get_img_from_file(img_file, resize_shape, img_shape)
                            cur_img_batch.append(cur_img)

                        img_x.append(np.squeeze(cur_img_batch))

                    if sns is not None:
                        sns_x.append(sns)

                    y.append(activity_dict_vuzix().index(activity))

                    if len(img_x) == batch_size or len(sns_x) == batch_size:
                        #print(img)

                        if type == "img":
                            yield np.array(img_x), np.eye(20)[np.array(y).astype(int)]

                        if type == "sns":
                            if len(sns_x) < batch_size:
                                print("Empty Sensor Bath")
                            yield np.array(sns_x), np.eye(20)[np.array(y).astype(int)]

                        img_x,sns_x, y = ([],[], [])

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


if __name__=="__main__":
    dataset = VuzixDataset()
    #dataset.extract_files()
    #dataset.assign_labels()
    #dataset.split_test_train()
    generator = VuzixDataset.flow_from_dir()
    while True:
        next(generator)