import os
import json
import math
import csv
import statistics
from sklearn.metrics import f1_score

DEBUGGING = False

# Function for reading from a ADL file
# Files are small enough to fit in memory, so just put all into same data structure
def read_adl_file(data_filepath):

    adl_data = {}  # Dict of date: [(start time, end time, event type) .. ]

    with open(data_filepath, "r") as csvfile:
        filereader = csv.reader(csvfile, delimiter='\t')
        next(filereader) # Skip header rows
        next(filereader)
        for row in filereader:

            row = [x for x in row if len(x)]
            # print(row)

            date = row[0].split()[0]

            if date not in adl_data.keys():
                adl_data[date] = []

            start_time = row[0].split()[1]
            end_time = row[1].split()[1]
            label = row[2]

            # Get digital format
            adl_event = (start_time, end_time, label)

            adl_data[date].append(adl_event)

    return adl_data


# Given a prompt index, figure out what gt labels we want selected.
def match_gt_label(prompt_index):

    gt_labels = []
    if int(prompt_index) == 1:
        gt_labels = ["Showering", "Grooming"]
    elif int(prompt_index) == 2:
        gt_labels = ["Spare_Time/TV", "Sleeping"]
    elif int(prompt_index) == 3:
        gt_labels = ["Leaving"]

    return gt_labels



# Get specific entries from chatgpt
def get_time_entries(response):

    time_entries = []

    # First, split by parentheses and get the responses
    entries = response.split("(")[1:]
    entries = [x.split(")")[0] for x in entries]

    # We have to make a slight fix - we ignore 'toilet' events because there's notion of
    #  when the event actually begins - we only get the 'ending' as a flush, resulting in incredibly
    #  poor performance.
    #  While it can be argued that this makes for an unfair evaluation, it's also worth
    #   pointing out that the objective of the system is to protect sensitive events,
    #  such as flush, and not 'toileting' events because the latter is not known
    #   to the sensing infrastructure anyway since no other modality can identify
    #   this event as occurring.  In other words, there is a discrepancy between
    #   what the user is aware of (e.g. a 'toileting' event), and what the sensor
    #  infrastructure is aware of (the flush event).  It doesn't make sense to
    #   protect events which only the user is able to identify (and not the sensing infrastructure,
    #    and as a result, any service provider which uses this data).
    entries = [x for x in entries if "Toilet" not in x]

    # For every entry, only get the times in the form [(start, end),...]
    entries = [x.split(",") for x in entries]
    entries = [x[:2] for x in entries]
    # Make sure that every entry has some numeric characters representing date and time
    entries = [x for x in entries if x[0][3] == ':']


    # For every time entry, be sure to strip it and conver to string
    for time_pair in entries:
        t1 = time_pair[0].strip()[1:-1]
        t2 = time_pair[1].strip()[1:-1]

        time_entries.append((t1, t2))

    
    return time_entries


# Parse ChatGPT responses - only get the intervals of interest
def parse_chatgpt_responses(result_file, num_dates):

    out_results = []  # List of [(start time, end time) .. ], where each 
                                # inner list represents a different date

    with open(result_file, "r") as datafile:
        
        response_data = datafile.read()
        response_data = response_data.split("***reply***")
        response_data = [x.split("***sent***")[0] for x in response_data]

        # First 3 parts are just command context
        for response in response_data[2:num_dates+2]:

            time_entries = get_time_entries(response)
            out_results.append(time_entries)

    return out_results


# Parse a sequence of intervals and combine them if necessary:
#  Remove intervals which have 0 time.
def combine_intervals(entries):

    out_entries = []
    current_start = entries[0][0]
    current_end = entries[0][1]

    # Combine entries
    for x in entries[1:]:
        if x[0] < current_end:
            current_end = x[1]
        else:  
            out_entries.append((current_start, current_end))
            current_start = x[0]
            current_end = x[1]
    out_entries.append((current_start, current_end))

    # Remove entries which have errors (e.g. 0 time diff)
    out_entries_final = []
    for x in out_entries:
        if x[0] < x[1]:
            out_entries_final.append(x)

    return out_entries_final


# Convert a hh:mm:ss time into seconds
def convert_to_seconds(entry):

    h, m, s = entry.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)


# Calculate IoU for two tuples
def measure_iou(tup1, tup2):

    # Convert time to seconds
    tup1 = (convert_to_seconds(tup1[0]), convert_to_seconds(tup1[1]))
    tup2 = (convert_to_seconds(tup2[0]), convert_to_seconds(tup2[1]))

    # Measure Iou
    intersection = [ max([tup1[0], tup2[0]]), min([tup1[1], tup2[1]]) ]
    intersection = max(0, intersection[1] - intersection[0])
    union = [ min([tup1[0], tup2[0]]), max([tup1[1], tup2[1]]) ]
    union = union[1] - union[0]

    iou = intersection / union
    return iou


# Calculate MAE for two tuples
#  Measures MAE in seconds
def measure_mae(tup1, tup2):

    # Convert time to seconds
    tup1 = (convert_to_seconds(tup1[0]), convert_to_seconds(tup1[1]))
    tup2 = (convert_to_seconds(tup2[0]), convert_to_seconds(tup2[1]))

    start_error = abs(tup1[0] - tup2[0])
    end_error = abs(tup1[1] - tup2[1])

    mae = (start_error + end_error) / 2
    return mae

# In the case of prompt-index 3, we do need to combine activities together
#  One can make a similar argument for the 'toilet' case described previously, where
#   we need to draw a distinction between the 'event' of leaving and the actual 
#   interval of leaving - we can only protect the former.  This is the main reason
#    why we modify response data, though it may seem unfair.  It's not so much
#   that the modification is unfair, but rather the underlying ground truth comparison
#   doesn't quite fit the problem, so we are adjusting for that.
def combine_leaving(response_entries):
    
    out_response_entries = []
    for i in range(0, len(response_entries)-1):
        leave_start = response_entries[i]
        leave_end = response_entries[i+1]

        # We only perform the append if the time is within a certain amount
        #  Let's say 2 hours
        if convert_to_seconds(leave_end[0]) - convert_to_seconds(leave_start[0]) < 18000:
            out_response_entries.append((leave_start[0], leave_end[1]))


    return out_response_entries


# Get timestamps from interval
def timestamps_from_interval(tup):

    start_time = convert_to_seconds(tup[0])
    end_time = convert_to_seconds(tup[1])

    return (start_time, end_time)


# Check if a time is within a particular range
def check_in_range(ts, tups):

    in_range = False
    for tup in tups:
        if ts >= tup[0] and ts <= tup[1]:
            in_range = True
            break
    return in_range


# Measure the f1 score by discretizing our time space
def measure_f1(response_entries, gt_entries):

    # First, convert each response entry and gt_entry into list of (start_time, end_time)
    response_timestamps = []
    gt_timestamps = []
    for response_entry in response_entries:
        response_timestamps.append(timestamps_from_interval(response_entry))
    
    for gt_entry in gt_entries:
        gt_timestamps.append(timestamps_from_interval(gt_entry))

    # Next, get the whole list of timestamps involving both gt and response
    relevant_timestamps = []
    start_t = min(response_timestamps[0][0], gt_timestamps[0][0])
    end_t = max(response_timestamps[-1][1], response_timestamps[-1][1])
    for i in range(start_t, end_t+1):

        # Check if this is inbetween any of either the gt timestamps or response timestamps
        if check_in_range(i, response_timestamps) or check_in_range(i, gt_timestamps):
            relevant_timestamps.append(i)
    
    
    

    # Now, create a list of one hot vectors for response and gt
    response_vector = [1 if check_in_range(j, response_timestamps) else 0 for j in relevant_timestamps]
    gt_vector = [1 if check_in_range(j, gt_timestamps) else 0 for j in relevant_timestamps]

    # print("Response Vector: ", response_vector)
    # print("gt_vector: ", gt_vector)

    # Now measure the f1
    # if len(relevant_timestamps) == 0:  # If no relevant timestamps, this is perfect.

    score = f1_score(gt_vector, response_vector, zero_division=1.0)

    if DEBUGGING:
        print("Len of timestamps: ", len(relevant_timestamps))
        print("Len of response vector: ", len(response_vector))
        print("Len of gt vector: ", len(gt_vector))
        print("F1: ", score)

    return score

# Measure level of matchup between the gt and the chatgpt response
def measure_matchup(gt_entries, response_entries, labels, prompt_index):

    # Combine intervals in response_entries
    # response_entries = combine_intervals(response_entries)
    if int(prompt_index) == 3:
        response_entries = combine_leaving(response_entries)

    # First, get the times of interest based on the labels for the gt
    parsed_gt_entries = []
    for gt_entry in gt_entries:
        if gt_entry[2] in labels:
            parsed_gt_entries.append((gt_entry[0], gt_entry[1]))
    # Do a bit of error checking
    parsed_gt_entries = [(x[0], x[1]) for x in parsed_gt_entries if x[1] > x[0]]

    if DEBUGGING:
        print("GT Entries:", parsed_gt_entries)
        print("Response Entries: ", response_entries)

    mae_scores = []
    iou_scores = []

    # Measure f1 score
    f1 = 0.0
    if len(parsed_gt_entries):
        f1 = measure_f1(response_entries, parsed_gt_entries)

    # Now, loop through every entry in the ground truth
    #  Measure the IoU and MAE in the interval
    for g_i, gt_entry in enumerate(parsed_gt_entries):


        current_best_iou = 0.0
        current_best_mae = 100000

        for response_entry in response_entries:

            # Measure IoU and mae between gt_entry and response 
            iou = measure_iou(response_entry, gt_entry)
            if iou > current_best_iou:
                current_best_iou = iou
            mae = measure_mae(response_entry, gt_entry)
            if mae < current_best_mae:
                current_best_mae = mae
        
        # print()
        # print(gt_entry)
        # print("IOU", current_best_iou)
        # print("MAE", current_best_mae)
        iou_scores.append(current_best_iou)
        mae_scores.append(current_best_mae)
    
    return mae_scores, iou_scores, f1




# get the filename and prompt index from the filename
def get_file_info(result_file):

    gt_name = result_file.split("_")[-2] + "_ADLs.txt"
    prompt_index = result_file.split("_")[-3]

    return gt_name, prompt_index


def analyze_results(gt_dir):

    # First, get all the result files that we need
    result_dir = "results" 
    
    result_files = os.listdir(result_dir)

    # Get all the 'query privacy states' files
    result_files = [x for x in result_files if "privacy_states" in x]

    if DEBUGGING:
        result_files = [result_files[-2]]

    # Iterate through each result file
    for result_file in result_files:

        print()
        print(result_file)

        # Get mae scores and iou scores
        all_mae_scores = []
        all_iou_scores = []
        all_f1_scores = []

        # Get the relevant ground_truth files
        gt_name, prompt_index = get_file_info(result_file)
        gt_filepath = gt_dir + gt_name
        # Get the ground truth data
        adl_data = read_adl_file(gt_filepath)
        # Get the relevant adl labels
        gt_labels_to_monitor = match_gt_label(prompt_index)

        # Parse the chatGPT responses
        result_filepath = os.path.join(result_dir, result_file)
        response_data = parse_chatgpt_responses(result_filepath, len(adl_data.keys()))

        # Make sure we have same number of responses as adl gt entries
        # print(len(response_data))
        # print(len(adl_data.keys()))
        # print(response_data[0])
        # print(sorted(adl_data.keys()))
        assert(len(response_data) == len(adl_data.keys()))

        # Now, iterate through every date
        for d_i, date  in enumerate(sorted(adl_data.keys())):

            # Get the relevant ground truth data
            gt_adl_data = adl_data[date]

            # Now, get the same data but for the ChatGPT reply
            current_response_data = response_data[d_i]

            mae_scores, iou_scores, f1 = \
                measure_matchup(gt_adl_data, current_response_data, gt_labels_to_monitor, \
                    prompt_index)
            
            all_mae_scores.extend(mae_scores)
            all_iou_scores.extend(iou_scores)
            all_f1_scores.append(f1)

            if DEBUGGING:
                print(date)
                print("MAE Scores: ", mae_scores)
                print("IOU Scores: ", iou_scores)
                print("F1 Score: ", f1)
                print("\nhere.")
            # print("Average MAE: ", statistics.mean(mae_scores))
            # print("Average IoU: ", statistics.mean(iou_scores))
            # print()

    
        print("Average MAE: ", statistics.mean(all_mae_scores))
        print("Average IoU: ", statistics.mean(all_iou_scores))
        print("Average F1: ", statistics.mean(all_f1_scores))
        print("Median MAE: ", statistics.median(all_mae_scores))
        print("Median IoU: ", statistics.median(all_iou_scores))
        print("Median F1: ", statistics.median(all_f1_scores))