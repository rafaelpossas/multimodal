"""
After moving all the files using the 1_ file, we run this one to extract
the images from the videos and also create a data file we can use
for training and testing later.
"""
import glob
import os
import os.path
import cv2
from subprocess import call
import h5py
import numpy as np
from sklearn.utils import shuffle

activity_dict = {
    'act01': (0, 'walking'), 'act02': (1, 'walking upstairs'), 'act03': (2, 'walking downstairs'),
    'act04': (3, 'riding elevator up'), 'act05': (4, 'riding elevator down'), 'act06': (5, 'riding escalator up'),
    'act07': (6, 'riding escalator down'), 'act08': (7, 'sitting'), 'act09': (8, 'eating'), 'act10': (9, 'drinking'),
    'act11': (10, 'texting'), 'act12': (11, 'phone calls'), 'act13': (12, 'working on pc'), 'act14': (13, 'reading'),
    'act15': (14, 'writing sentences'), 'act16': (15, 'organizing files'), 'act17': (16, 'running'),
    'act18': (17, 'push-ups'),
    'act19': (18, 'sit-ups'), 'act20': (19, 'cycling')
}


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

def create_dataset(chunk_size, step_size):
    folder = '../multimodal_dataset/video_images/'
    x = []
    y = []
    for path, subdirs, files in os.walk(folder):
        if len(subdirs) > 0:
            print("Current Path: " + path)
        for dir in subdirs:
            cur_class = []
            cur_image = []

            files = glob.glob(path+"/" + dir + '/*.jpg')

            if len(files) > 0:
                print("Creating images for: "+dir)

            for ix, name in enumerate(files):
                if ix < 450:
                    cur_image = cv2.resize(cv2.imread(name), (224, 224))
                    cur_class.append(cur_image)

            while 0 < len(cur_class) < 450:
                cur_class.append(cur_image)

            cur_class = greedy_split(np.array(cur_class), chunk_size=chunk_size, step_size=step_size)

            if len(cur_class) > 0:
                for cur_frame in cur_class:
                    x.append(cur_frame)
                    y.append(activity_dict[path.split('/')[-1]][0])

    x, y = shuffle(x, y)

    with h5py.File("activity_frames.hdf5", "w") as hf:
        hf.create_dataset("x", data=x)
        hf.create_dataset("y", data=y)

        # if len(x) > 0:
        #     print(min([len(p) for p in x]))
        # x = []
        # y = []
    return x, y


def extract_files():
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
    folders = ['../multimodal_dataset/video']

    for folder in folders:
        class_folders = glob.glob(folder + '*')

        for vid_class in class_folders:
            class_files = glob.glob(vid_class + '/*.mp4')

            for video_path in class_files:
                # Get the parts of the file.
                video_parts = get_video_parts(video_path)

                filename, class_label, seq, video_dir = video_parts

                # Only extract if we haven't done it yet. Otherwise, just get
                # the info.
                cur_class_dir = video_dir + class_label + '/'
                if not bool(os.path.exists(cur_class_dir+'/'+seq)):
                    os.makedirs(cur_class_dir+'/'+seq)

                src = video_dir + '/' + filename
                dest = cur_class_dir + '/' + seq + '/' + class_label+'_'+seq+'_%04d.jpg'
                call(["ffmpeg", "-i", src, dest])

                # Now get how many frames it is.
                #nb_frames = get_nb_frames_for_video(video_parts)

                #data_file.append([train_or_test, classname, filename_no_ext, nb_frames])

                #print("Generated %d frames for %s" % (nb_frames, filename_no_ext))

    # with open('data_file.csv', 'w') as fout:
    #     writer = csv.writer(fout)
    #     writer.writerows(data_file)

    print("Extracted and wrote %d video files." % (len(data_file)))

def get_nb_frames_for_video(video_parts):
    """Given video parts of an (assumed) already extracted video, return
    the number of frames that were extracted."""
    train_or_test, classname, filename_no_ext, _ = video_parts
    generated_files = glob.glob(train_or_test + '/' + classname + '/' +
                                filename_no_ext + '*.jpg')
    return len(generated_files)

def get_video_parts(video_path):
    """Given a full path to a video, return its parts."""
    parts = video_path.split('/')
    filename = parts[-1]
    dir = '/'.join(parts[0:-1]) + '/'
    class_label = filename[0:5]
    seq = filename[5:10]

    return filename, class_label, seq, dir


def main():
    """
    Extract images from videos and build a new file that we
    can use as our data input file. It can have format:
    [train|test], class, filename, nb frames
    """
    create_dataset(10, 10)

if __name__ == '__main__':
    main()