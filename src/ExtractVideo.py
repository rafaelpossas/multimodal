"""
After moving all the files using the 1_ file, we run this one to extract
the images from the videos and also create a data file we can use
for training and testing later.
"""
import csv
import glob
import os
import os.path
import cv2
from subprocess import call
import matplotlib.pyplot as plt


def create_dataset():
    folder = '../multimodal_dataset/video_images/'
    x = []
    y = []
    for path, subdirs, files in os.walk(folder):
        for dir in subdirs:
            cur_class = []
            for name in glob.glob(path+"/" + dir + '/*.jpg'):
                cur_image = cv2.resize(cv2.imread(name), (224, 224))
                plt.imshow(cur_image)
                cur_class.append(cur_image)
            if len(cur_class) > 0:
                x.append(cur_class)
                y.append(path.split('/')[-1])
        # if len(x) > 0:
        #     print(min([len(p) for p in x]))
        # x = []
        # y = []

    print(min(len(p) for p in x))



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
    create_dataset()

if __name__ == '__main__':
    main()