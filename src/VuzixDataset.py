import glob
from subprocess import call
import os
import numpy as np
import pandas as pd
import cv2
import h5py
import shutil




class VuzixDataset:

    activity_dict = ['walking', 'walking down/upstairs', 'chopping food', 'riding elevator', 'brushing teeth',
                     'riding escalator', 'talking with people', 'watching tv', 'eating', 'cooking on stove',
                     'browsing mobile phone', 'washing dishes', 'working on pc', 'reading', 'writing',
                     'lying down', 'running', 'doing push ups', 'doing sit ups', 'cycling']

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

    def assign_labels(self, folders=['vuzix/train/','vuzix/test/'],
                      sensor_files=['ACC', 'GYR'], labels_file_name='labels'):

        for folder in sorted(folders):
            class_folders = glob.glob(folder + '*')
            for global_ix, vid_class in enumerate(sorted(class_folders)):
                label_array = None
                sensors_array = None
                print(vid_class)
                all_img_files = glob.glob(vid_class + '/images/*.jpg')

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
                real_labels_array = [activity_dict.index(cur_labels[int(value)].rstrip()) for value in label_array]
                sensors_array = np.column_stack((sensors_array, real_labels_array))

                np.save(os.path.join(vid_class, "acc_gyr"), sensors_array)

                for file in all_img_files:
                    cur_index = int(int(file.split("_")[-1].split(".")[0]))
                    dest_dir = None
                    try:
                        dest_dir = os.path.join(vid_class, "images", "_".join([str(label_array[cur_index-1]),
                                                                               activity_dict[real_labels_array[cur_index-1]]
                                                                              .replace("/", "_")]))
                    except IndexError:
                        os.remove(file)
                        file = None

                    finally:

                        if not os.path.exists(dest_dir):
                            os.mkdir(dest_dir)

                        if file is not None:
                            shutil.move(file, dest_dir)

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
                    dataset = np.random.choice(['train', 'test'], p=[0.8, 0.2])

                    if dataset == "test":
                        prefix = "test"
                    else:
                        prefix = "train"

                    src = os.path.join(folder, video_dir, filename)
                    dest = os.path.join(folder, prefix, video_dir, "images")

                    if not bool(os.path.exists(os.path.join(folder, prefix))):
                        os.makedirs(os.path.join(folder, prefix))

                    if not bool(os.path.exists(dest)):
                        os.makedirs(dest)

                    dest = os.path.join(dest, filename.split('.')[0] + '_%04d.jpg')

                    call(["ffmpeg", "-i", src, dest])

                    txt_files = glob.glob(os.path.join(folder, video_dir, '*.txt'))
                    for file in txt_files:
                        shutil.copy(file, os.path.join(folder, prefix, video_dir))

                    # Now get how many frames it is.
                    # nb_frames = get_nb_frames_for_video(video_parts)

                    # data_file.append([train_or_test, classname, filename_no_ext, nb_frames])

                    # print("Generated %d frames for %s" % (nb_frames, filename_no_ext))

        # with open('data_file.csv', 'w') as fout:
        #     writer = csv.writer(fout)
        #     writer.writerows(data_file)

        print("Extracted and wrote %d video files." % (len(data_file)))
