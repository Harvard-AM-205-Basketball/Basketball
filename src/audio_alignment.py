"""
Citation: This file was taken "as is" from the project VideoSync by Allison Deal.
https://github.com/allisonnicoledeal/VideoSync/blob/master/alignment_by_row_channels.py
Minimal changes were made to:
    (1) Work on Python 3
    (2) Take wav files as input
"""

import scipy.io.wavfile
import numpy as np
# from subprocess import call
import math
from tqdm import tqdm

# Extract audio from video file, save as wav auido file
# INPUT: Video file
# OUTPUT: Does not return any values, but saves audio as wav file
#    def extract_audio(dir, video_file):
#        track_name = video_file.split(".")
#        audio_output = track_name[0] + "WAV.wav"  # !! CHECK TO SEE IF FILE IS IN UPLOADS DIRECTORY
#        output = dir + audio_output
#        call(["avconv", "-y", "-i", dir+video_file, "-vn", "-ac", "1", "-f", "wav", output])
#        return output


# Read file
# INPUT: Audio file
# OUTPUT: Sets sample rate of wav file, Returns data read from wav file (numpy array of integers)
def read_audio(audio_file):
    rate, data = scipy.io.wavfile.read(audio_file)  # Return the sample rate (in samples/sec) and data from a WAV file
    return data, rate


def make_horiz_bins(data, fft_bin_size, overlap, box_height):
    # print(f'make_horiz_bins...')
    horiz_bins = {}
    # process first sample and set matrix height
    sample_data = data[0:fft_bin_size]  # get data for first sample
    if (len(sample_data) == fft_bin_size):  # if there are enough audio points left to create a full fft bin
        intensities = fourier(sample_data)  # intensities is list of fft results
        for i in range(len(intensities)):
            box_y = i/box_height
            # if horiz_bins.has_key(box_y):
            if box_y in horiz_bins:
                horiz_bins[box_y].append((intensities[i], 0, i))  # (intensity, x, y)
            else:
                horiz_bins[box_y] = [(intensities[i], 0, i)]
    # process remainder of samples
    x_coord_counter = 1  # starting at second sample, with x index 1
    for j in tqdm(range(int(fft_bin_size - overlap), len(data), int(fft_bin_size-overlap))):
    # for j in range(int(fft_bin_size - overlap), len(data), int(fft_bin_size-overlap)):
        sample_data = data[j:j + fft_bin_size]
        if (len(sample_data) == fft_bin_size):
            intensities = fourier(sample_data)
            for k in range(len(intensities)):
                box_y = k/box_height
                # if horiz_bins.has_key(box_y):
                if box_y in horiz_bins:
                    horiz_bins[box_y].append((intensities[k], x_coord_counter, k))  # (intensity, x, y)
                else:
                    horiz_bins[box_y] = [(intensities[k], x_coord_counter, k)]
        x_coord_counter += 1

    return horiz_bins


# Compute the one-dimensional discrete Fourier Transform
# INPUT: list with length of number of samples per second
# OUTPUT: list of real values len of num samples per second
def fourier(sample):  #, overlap):
    mag = []
    fft_data = np.fft.fft(sample)  # Returns real and complex value pairs
    for i in range(len(fft_data)//2):
        # r = fft_data[i].real**2
        # j = fft_data[i].imag**2
        r = np.sum(fft_data[i].real**2)
        j = np.sum(fft_data[i].imag**2)
        mag.append(round(math.sqrt(r+j),2))

    return mag


def make_vert_bins(horiz_bins, box_width):
    # print(f'make_vert_bins...')
    boxes = {}
    for key in tqdm(horiz_bins.keys()):
        for i in range(len(horiz_bins[key])):
            box_x = horiz_bins[key][i][1] / box_width
            # if boxes.has_key((box_x,key)):
            if (box_x,key) in boxes:
                boxes[(box_x,key)].append((horiz_bins[key][i]))
            else:
                boxes[(box_x,key)] = [(horiz_bins[key][i])]

    return boxes


def find_bin_max(boxes, maxes_per_box):
    freqs_dict = {}
    for key in tqdm(boxes.keys()):
        max_intensities = [(1,2,3)]
        for i in range(len(boxes[key])):
            if boxes[key][i][0] > min(max_intensities)[0]:
                if len(max_intensities) < maxes_per_box:  # add if < number of points per box
                    max_intensities.append(boxes[key][i])
                else:  # else add new number and remove min
                    max_intensities.append(boxes[key][i])
                    max_intensities.remove(min(max_intensities))
        for j in range(len(max_intensities)):
            # if freqs_dict.has_key(max_intensities[j][2]):
            if max_intensities[j][2] in freqs_dict:
                freqs_dict[max_intensities[j][2]].append(max_intensities[j][1])
            else:
                freqs_dict[max_intensities[j][2]] = [max_intensities[j][1]]

    return freqs_dict


def find_freq_pairs(freqs_dict_orig, freqs_dict_sample):
    time_pairs = []
    for key in tqdm(freqs_dict_sample.keys()):  # iterate through freqs in sample
        if key in freqs_dict_orig:  # if same sample occurs in base
            for i in range(len(freqs_dict_sample[key])):  # determine time offset
                for j in range(len(freqs_dict_orig[key])):
                    time_pairs.append((freqs_dict_sample[key][i], freqs_dict_orig[key][j]))

    return time_pairs


def find_delay(time_pairs):
    t_diffs = {}
    for i in tqdm(range(len(time_pairs))):
    # for i in range(len(time_pairs)):
        delta_t = time_pairs[i][0] - time_pairs[i][1]
        if t_diffs.has_key(delta_t):
            t_diffs[delta_t] += 1
        else:
            t_diffs[delta_t] = 1
    t_diffs_sorted = sorted(t_diffs.items(), key=lambda x: x[1])
    print(t_diffs_sorted)
    time_delay = t_diffs_sorted[-1][0]

    return time_delay


# Find time delay between two video files
def align(wavfile1, wavfile2, fft_bin_size=1024, overlap=0, box_height=512, box_width=43, samples_per_box=7):
    # Process first file
    # wavfile1 = extract_audio(dir, video1)
    raw_audio1, rate = read_audio(wavfile1)
    print(f'make_horiz_bins on {wavfile1}...')
    bins_dict1 = make_horiz_bins(raw_audio1[:44100*120], fft_bin_size, overlap, box_height) #bins, overlap, box height
    print(f'make_vert_bins on {wavfile1}...')
    boxes1 = make_vert_bins(bins_dict1, box_width)  # box width
    print(f'find_bin_max on {wavfile1}...')
    ft_dict1 = find_bin_max(boxes1, samples_per_box)  # samples per box

    # Process second file
    # wavfile2 = extract_audio(dir, video2)
    raw_audio2, rate = read_audio(wavfile2)
    print(f'make_horiz_bins on {wavfile2}...')
    bins_dict2 = make_horiz_bins(raw_audio2[:44100*60], fft_bin_size, overlap, box_height)
    print(f'make_vert_bins on {wavfile2}...')
    boxes2 = make_vert_bins(bins_dict2, box_width)
    print(f'find_bin_max on {wavfile2}...')
    ft_dict2 = find_bin_max(boxes2, samples_per_box)

    # Determine time delay
    print(f'find_freq_pairs...')
    pairs = find_freq_pairs(ft_dict1, ft_dict2)
    print(f'find_delay...')
    delay = find_delay(pairs)
    samples_per_sec = float(rate) / float(fft_bin_size)
    seconds= round(float(delay) / float(samples_per_sec), 4)

    if seconds > 0:
        return (seconds, 0)
    else:
        return (0, abs(seconds))
    
wavfile1 = r'../audio/Camera1.wav'
wavfile2 = r'../audio/Camera2.wav'

gap = align(wavfile1, wavfile2, fft_bin_size=256, box_height=64, box_width=8, samples_per_box=2)
