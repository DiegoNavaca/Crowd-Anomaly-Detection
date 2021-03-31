import os
import glob
import time
import pickle

import cv2 as cv

from descriptors import extract_descriptors

# Extracts the descriptors of all files in a directory and saves them in separate files
def extract_Descriptors_Dir(params, input_dir, output_dir, gt, verbose = 1, video_classification = False, skip_extraction = False):
    # The descriptors are stored in a .data file in the output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    start = time.time()
    
    for i, video in enumerate(glob.glob(input_dir+"**", recursive = True)):
        try:
            # Check if file is a video
            assert  video.split("/")[-1][-4:]  in (".avi", ".mp4")
            
            # Name of the file without extension
            name = video.split("/")[-1][:-4]

            data_file = output_dir+name+".data"
            if not skip_extraction:
                if verbose:
                    print("{} - {}".format(i,video))
                if os.path.exists(data_file):
                    os.remove(data_file)

                if "skip_to_middle" in params["others"]:
                    cap = cv.VideoCapture(video)
                    params["others"]["skip_frames"] = int(cap.get(cv.CAP_PROP_FRAME_COUNT)) // 2 - params["L"]

                video_start = time.time()
                extract_descriptors(video, params["L"], params["t1"], params["t2"], params["min_motion"],
                                    params["fast_threshold"], data_file, others = params["others"])
                if verbose:
                    print("Tiempo extraccion: {:1.3f} / {:1.3f}".format(time.time()-video_start, time.time()-start))

            # We also store the labels in a .labels file
            if video_classification:
                labels = [gt[name]]
            else:
                try:
                    beginning, end, length = gt[name]
                    L = params["L"]
                    labels = [1 if i < (beginning-L) or i > (end-L) else -1 for i in range(length-L+1)]
                except TypeError:
                    labels = [gt[name]]
            
            labels_file = open(output_dir+name+".labels","wb")
            pickle.dump(labels,labels_file)
            labels_file.close()

        except AssertionError:
            if verbose and not skip_extraction:
                print("{} is not a video".format(video))

        
# Reads the ground truth from a file
def get_Ground_Truth(in_file):
    f = open(in_file,"r")
    gt = {}
    for line in f:
        lista = line.split(",")
        # name: (beginning anomaly, end anomaly, length video)
        if len(lista) > 2:
            gt[lista[0]] = (int(lista[1]),int(lista[2]), int(lista[3]))
        else:
            gt[lista[0]] = int(lista[1])

    f.close()
        
    return gt

# Separates the videos into their classes
def get_Classes(in_file):
    f = open(in_file,"r")
    classes = {}
    for line in f:
        lista = line.split(",")
        
        if int(lista[1]) not in classes:
            classes[int(lista[1])] = []

        classes[int(lista[1])].append(lista[0])
        
    f.close()

    return classes

# Reads the labels from a file
def read_Labels(labels_file):
    f = open(labels_file,"rb")
    labels = pickle.load(f)
    f.close()

    return labels
