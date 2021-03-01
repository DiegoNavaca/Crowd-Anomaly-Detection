import os
import glob
import time
import pickle

import numpy as np

from descriptors import extract_descriptors

# Extracts the descriptors of all files in a directory and saves them in separate files
def extract_Descriptors_Dir(params, input_dir, output_dir, gt, verbose = True, video_classification = False):
    # The descriptors are stored in a .data file in the output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i, video in enumerate(glob.glob(input_dir+"**", recursive = True)):
        if verbose:
            print("{} - {}".format(i,video))
        try:
            # Check if file is a video
            assert  video.split("/")[-1][-4:]  in (".avi", ".mp4")
            
            # Name of the file without extension
            name = video.split("/")[-1][:-4]

            data_file = output_dir+name+".data"
            # if os.path.exists(data_file):
            #    os.remove(data_file)
            
            # extract_descriptors(video, params["L"], params["t1"], params["t2"], params["min_motion"],
            #                     params["fast_threshold"], params["max_num_features"], data_file)

            # We also store the labels in a .labels file
            if video_classification:
                labels = [gt[name]]
            else:
                beginning, end, length = gt[name]
                L = params["L"]
                labels = [1 if i < (beginning-L) or i > (end-L) else -1 for i in range(length-L+1)]
            
            labels_file = open(output_dir+name+".labels","wb")
            pickle.dump(labels,labels_file)
            labels_file.close()

        except AssertionError:
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

# Reads the labels from a file (list format)
def read_Labels(labels_file):
    f = open(labels_file,"rb")
    labels = pickle.load(f)
    f.close()

    return labels
