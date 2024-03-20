import os
import json
import math
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.transforms
import matplotlib.pyplot as plt
import itertools



# Given a string, figure out the middle space and create a new line
def create_newline_in_statement(statement):

    split_s = statement.split(" ")
    half_index = len(split_s)//2
    p1 = ' '.join(split_s[:half_index+1])
    p2 = ' '.join(split_s[half_index+1:])
    out = p1 + "\n" + p2

    return out


def plot_confusion_matrix(cm,
                          target_names1,
                          target_names2,
                          title='',
                          cmap=None,
                          normalize=True):
    # accuracy = np.trace(cm) / np.sum(cm).astype('float')
    # misclass = 1 - accuracy

    # if cmap is None:
    cmap = plt.get_cmap('RdBu')

    # fig = plt.figure(figsize=(10, 7))
    fig = plt.figure(figsize=(7, 5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(target_names1))
    target_names1 = [create_newline_in_statement(x) for x in target_names1]
    target_names2 = [create_newline_in_statement(x) for x in target_names2]
    plt.xticks(tick_marks, target_names1, rotation=45, ha='right')
    plt.yticks(tick_marks, target_names2)



    plt.tight_layout()
    plt.savefig("nl.png")

# # Open the study log
# study_log_filepath = "results/user_study_0.log"
def analyze_results(study_log_filepath):

    # Read all lines
    with open(study_log_filepath, "r") as f:

        all_data = f.read()
        # Split by sent
        all_data = all_data.split("***sent***")

        result_dict = {} # This stores tuples of (principles, recipient) as keys, and scores

        # Iterate through each line
        for entry in all_data[2:]:

            # Get the principles and recipient of this entry
            principles = entry.split("Transmission principles:")[1].split(".")[0].strip()
            recipient = entry.split("Recipient:")[1].split("\n")[0].strip()


            print(entry)

            score = 0
            if "score==**" in entry or "Score==**" in entry:
                # Get the score
                score = entry.split("==**")[1].split("**")[0]
            elif "score==" in entry or "Score==" in entry:
                score = entry.split("==")[1].split("\n")[0]
            elif "acceptability score" in entry:
                score = entry.split("acceptability score of ")[1].split(" to")[0]
            else:
                score = entry.split("score of ")[1].split(" to")[0]
            score = float(score)

            dict_key = (principles, recipient)
            if dict_key not in result_dict:
                result_dict[dict_key] = [score]
            else:
                result_dict[dict_key].append(score)
        

    #  Now average the scores
    avg_scores = {}
    for key in result_dict:
        avg_scores[key] = sum(result_dict[key])/len(result_dict[key])

    # Now get the scores from the user study
    if not os.path.exists("results/gt_user_study"):
        gt_vals = {}
        for key in avg_scores:
            user_study_val = input(str(key) + ": ")
            gt_vals[str(key)] = float(user_study_val)

        # Save gt_vals to file
        with open("results/gt_user_study", "w") as f:
            json.dump(gt_vals, f)
    else:
        with open("results/gt_user_study", "r") as f:
            gt_vals = json.load(f)



    # Iterate and calculate the difference
    differences_dict = {}
    for key in gt_vals:

        gt_val = gt_vals[key]
        llm_val = avg_scores[eval(key)]

        # Check if these fall under the same binary acceptability level
        if math.copysign(1, gt_val) != math.copysign(1, llm_val):
            differences_dict[key] = (False, abs(abs(gt_val)-abs(llm_val)))
        else:
            differences_dict[key] = (True, abs(abs(gt_val)-abs(llm_val)))




    # Create the figure here...
    principles = {}  # {title : index}
    recipients = {}  # {title : index}
    for key in differences_dict:
        key_tup = eval(key)
        principle = key_tup[0]
        recipient = key_tup[1]

        if principle not in principles:
            principles[principle] = len(principles.keys())
        if recipient not in recipients:
            recipients[recipient] = len(recipients.keys())

    # Intialize our confusion matrix
    cf = np.zeros((len(principles.keys()), len(recipients.keys())))
    for key in differences_dict:
        key_tup = eval(key)
        principle = key_tup[0]
        recipient = key_tup[1]

        # Get the indexes
        p_i = principles[principle]
        r_i = recipients[recipient]
        # value
        val = differences_dict[key][1]
        if not differences_dict[key][0]:
            val = -1*val
        
        cf[p_i][r_i] = val

    # Reorder our labels
    principle_labels = [(x,y) for x,y in principles.items()]
    principle_labels = sorted(principle_labels, key=lambda x: x[1])
    principle_labels = [x[0] for x in principle_labels]

    recipient_labels = [(x,y) for x,y in recipients.items()]
    recipient_labels = sorted(recipient_labels, key=lambda x: x[1])
    recipient_labels = [x[0] for x in recipient_labels]

    print(principle_labels)


    print(cf)

    # Now draw our figure
    plot_confusion_matrix(cf,
                            principle_labels,
                            recipient_labels)