import os
import librosa
import librosa.display as ld
import math
import json


DATASET_PATH = '../../Data/genres_orignal'
JSON_PATH = 'data.json'
DURATION = 30
SAMPLE_RATE = 22050
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

def save_mfcc(dataset_path, json_path,n_mels=10,n_fft=2048,hop_length=512,num_segments=5,sr=22050):

    # dictonary to store data 
    data = {
        'mapping': [],
        'mfcc': [],
        'label': [],
    }

    num_samples_per_segment = int(SAMPLES_PER_TRACK/num_segments)

    # ensuring that the length of the sample vector is same in all cases
    expected_num_mfccs_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)


    # looping through all the genres available in the dataset_path
    for i, (dir_path, dir_names, file_names) in enumerate(os.walk(dataset_path)):


        # ensuring we are not on the root level :
        if dir_path is not dataset_path:


            # saving the semantic labels 
            dir_path_components = dir_path.split('/')  # genres_original/blues +> [genres_original, blues]
            semantic_label = dir_path_components[-1]
            data['mapping'].append(semantic_label)
            print("\nPreprocessing for {}".fornat(semantic_label))



            # next we will go through all the files in curent path 
            # and process each files for the specific genre folder we are present
            for f in file_names:

                # load audio file
                # file names 'f' just gives the name not path so ,
                file_path = os.path.join(dir_path,f)
                audio, sr = librosa.load(file_path,sr=sr)



                # dividing the signal into multiple segments to get more input data
                # extracting mfccs
                # storing data
                for seg in range(num_samples_per_segment):
                    start_sample = num_samples_per_segment * seg
                    finish_sample = start_sample + num_samples_per_segment
                    mfccs = librosa.feature.mfcc(audio[start_sample:finish_sample],n_mels=n_mels,n_fft=n_fft,hop_length=hop_length)
                    mfccs = mfccs.T 

                    # ensuring that our data has same length of vectors per segments
                    if len(mfccs) == expected_num_mfccs_vectors_per_segment:
                        data['mfcc'].append(mfccs.tolist())
                        data['label'].append(i-1)

                        print("\n{}, segment:{}".format(file_path,seg))

    with open(json_path,'w') as fp:
        json.dump(data,fp, indent=5)



if __name__ == '__main__':
    save_mfcc(dataset_path=DATASET_PATH,json_path=JSON_PATH)

    