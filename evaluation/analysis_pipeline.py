from deepface import DeepFace
import os
from tqdm import tqdm
from sklearn.metrics import f1_score
import statistics
import cv2
import matplotlib.pyplot as plt

# Get privacy (age, gender, race) and utility information (emotion)
def get_all_info(deepface_info):

    age_preds = []
    gender_preds = []
    race_preds = []
    emotion_preds = []
    bbox_preds = []
    skipped = []
    # Iterate through every detected face and get all info
    for face_data in deepface_info:

        age_info = face_data["age"]
        gender_info = face_data["dominant_gender"]
        race_info = face_data["dominant_race"]
        emotion_info = face_data["dominant_emotion"]
        bbox_info = face_data["region"]

        # # Skip images if the confidence is not high
        # if face_data["face_confidence"] < 0.8:
        #     continue

        
        # Something weird is happening - whenever it fails to detect a face it 
        #  seems to default to a prediction that is the size of the frame
        if bbox_info['x'] == 0 and bbox_info['y'] == 0 and \
            bbox_info['w'] == 800 and bbox_info['h'] == 600:
            continue

        # Add to our output
        age_preds.append(age_info)
        gender_preds.append(gender_info)
        race_preds.append(race_info)
        emotion_preds.append(emotion_info)
        bbox_preds.append(bbox_info)


    return age_preds, gender_preds, race_preds, emotion_preds, bbox_preds


# Perform evaluation using deepface
def face_analysis(image_path):

    info = DeepFace.analyze(img_path=image_path, \
        actions = ['age', 'gender', 'race', 'emotion'], enforce_detection=False,\
            detector_backend='yolov8')
    analysis = get_all_info(info)
    return analysis


# Calculate the age group
#  Returns the index of the age group for a given age
def identify_age_group(age):
    age_groups = [[18,24], [25,34], [35,44], [45,54], [55,64], [65,100]]

    age_index = -1
    for a_i, age_group in enumerate(age_groups):
        if age >= age_group[0] and age <= age_group[1]:
            age_index = a_i
            break

    return age_index

# We need to do a comparison between two images
#  If the two images are the same, it means that our facefusion/blur mechanism
#  did not work.
def compare_images(baseline_bboxes, method, image_filenames, image_index):

    mechanism_failed = False

    # Get the filenames for the baseline and this method
    #  for this image
    baseline_image_path = image_filenames["nothing"][image_index]
    method_image_path = image_filenames[method][image_index]

    # Now, read in both images and compare their values for the given bbox locations
    baseline_image = cv2.cvtColor(cv2.imread(baseline_image_path), 
                        cv2.COLOR_BGR2RGB)
    method_image = cv2.cvtColor(cv2.imread(method_image_path), 
                        cv2.COLOR_BGR2RGB)
    for bbox in baseline_bboxes:

        baseline_image_box = baseline_image[bbox['y']:bbox['y']+bbox['h'],\
                                            bbox['x']:bbox['x']+bbox['w']]
        method_image_box = method_image[bbox['y']:bbox['y']+bbox['h'],\
                                            bbox['x']:bbox['x']+bbox['w']]
        
        if (baseline_image_box==method_image_box).all():
            mechanism_failed = True
            break

    return mechanism_failed

    


# Do the alignment for an image
#  For now keep it simple and just do a comparison by length instead of IoU
def align_for_images(baseline_preds, curr_method_preds, image_filenames, method):

    baseline_out = {"age":[], "gender":[], "race":[], "emotion":[]}
    curr_method_out = {"age":[], "gender":[], "race":[], "emotion":[]}

    # Iterate through every image
    for image_index in range(0, len(baseline_preds[0])):

    
        # Get the bboxes
        baseline_bboxes = baseline_preds[4][image_index]
        method_bboxes = curr_method_preds[4][image_index]

        # If the comparison shows that the mechanism did not work, we continue
        if compare_images(baseline_bboxes, method, image_filenames, image_index):
            continue

        # Loop through the maximum number of values
        for val_index in range(0, max(len(baseline_bboxes), len(method_bboxes))):

            # If we hit the max, we add more values
            if val_index >= len(baseline_bboxes):
                baseline_out["age"].append(-1)
                baseline_out["gender"].append("unknown")
                baseline_out["race"].append("unknown")
                baseline_out["emotion"].append("unknown")

                curr_method_age = identify_age_group(curr_method_preds[0][image_index][val_index])
                curr_method_out["age"].append(curr_method_age)
                curr_method_out["gender"].append(curr_method_preds[1][image_index][val_index])
                curr_method_out["race"].append(curr_method_preds[2][image_index][val_index])
                curr_method_out["emotion"].append(curr_method_preds[3][image_index][val_index])

            elif val_index >= len(method_bboxes):
                curr_method_out["age"].append(-1)
                curr_method_out["gender"].append("unknown")
                curr_method_out["race"].append("unknown")
                curr_method_out["emotion"].append("unknown")

                baseline_age = identify_age_group(baseline_preds[0][image_index][val_index])
                baseline_out["age"].append(baseline_age)
                baseline_out["gender"].append(baseline_preds[1][image_index][val_index])
                baseline_out["race"].append(baseline_preds[2][image_index][val_index])
                baseline_out["emotion"].append(baseline_preds[3][image_index][val_index])

            else:
                # Otherwise, append like normal
                curr_method_age = identify_age_group(curr_method_preds[0][image_index][val_index])
                curr_method_out["age"].append(curr_method_age)
                curr_method_out["gender"].append(curr_method_preds[1][image_index][val_index])
                curr_method_out["race"].append(curr_method_preds[2][image_index][val_index])
                curr_method_out["emotion"].append(curr_method_preds[3][image_index][val_index])

                baseline_age = identify_age_group(baseline_preds[0][image_index][val_index])
                baseline_out["age"].append(baseline_age)          
                baseline_out["gender"].append(baseline_preds[1][image_index][val_index])
                baseline_out["race"].append(baseline_preds[2][image_index][val_index])
                baseline_out["emotion"].append(baseline_preds[3][image_index][val_index])

    # At this point, curr_method_out and baseline out should have the same number of items
    assert(len(curr_method_out["age"]) == len(baseline_out["age"]))
    
    return baseline_out, curr_method_out



# Do some alignment between the baseline and the other methods
#  Method preds is a dict
def score_method_pairs(baseline, method_preds, image_filenames):
        

    out_results = {}
    
    # Now, compare it against every other method
    for method in method_preds.keys():
        # print("METHOD", method)
        aligned_baseline_results, aligned_method_results\
                 = align_for_images(baseline, method_preds[method], image_filenames, method)

        age_f1 = f1_score(aligned_baseline_results["age"], aligned_method_results["age"], average="macro")
        gender_f1 = f1_score(aligned_baseline_results["gender"], aligned_method_results["gender"], average="macro")
        race_f1 = f1_score(aligned_baseline_results["race"], aligned_method_results["race"], average="macro")
        emotion_f1 = f1_score(aligned_baseline_results["emotion"], aligned_method_results["emotion"], average="macro")

        # print(aligned_baseline_results["age"])
        # print(aligned_method_results["age"])
        # print(aligned_baseline_results["gender"])
        # print(aligned_method_results["gender"])

        # print(age_f1)
        # print(gender_f1)
        # print(race_f1)
        # print(emotion_f1)

        out_results[method] = {"age": age_f1, "gender": gender_f1, \
            "race": race_f1, "emotion": emotion_f1}

    return out_results

# Score a dict of different methods
def score_methods(result_dict, image_filenames):
    
    # To perform scoring, everything is just done by f1, except for age
    #  which is split into age groups
    
    # # Our baseline is the 'nothing' method, so everything is compared against it
    method_keys = [x for x in result_dict.keys() if x != "nothing"]

    # # Iterate through each method
    # for method in method_keys:
    
    # Get an index to operate with
    baseline = result_dict["nothing"]
    method_preds = {x:result_dict[x] for x in method_keys}

    method_f1_scores = score_method_pairs(baseline, method_preds, image_filenames)
    
    return method_f1_scores


# Now, run the deepface analysis on every image for all the relevant folders
#  in our 'processed_video'.
def analyze_results(processed_video_folder):

    # Get all the relevant folders
    folders_to_analyze = os.listdir(processed_video_folder)
    folders_to_analyze = [x for x in folders_to_analyze if "mp4" not in x]

    # Get the prefixes
    prefixes = ["_".join(x.split("_")[:-2]) for x in folders_to_analyze ]
    prefixes = list(set(prefixes))

    method_all_f1s = {}

    # Iterate through each prefix, folder, and then image
    for prefix in prefixes:

        # Keep track of the scores for this prefix
        prefix_results = {} # Keys are type of pipeline, values are ([age preds], [gender preds], etc)
        image_filenames = {}  # Keys are type of pipeline, values are [image file]

        # Get all folders which match this prefix
        curr_folders = [x for x in folders_to_analyze if prefix in x]

        for folder in curr_folders:
            curr_folder = os.path.join(processed_video_folder, folder)
            image_files = [os.path.join(curr_folder, x) for x in os.listdir(curr_folder)]
            image_files = sorted(image_files, key=lambda x : int(x.split("frame")[-1].split(".")[0]))



            # Get the current type of pipeline and create an entry in prefix_results
            pipeline_type = folder.split("_")[-1]

            # if pipeline_type != "nothing" and pipeline_type != "blur":
            #     continue

            prefix_results[pipeline_type] = ([], [], [], [], [])
            
            

            for image_filepath in tqdm(image_files):
                # print(image_filepath)
                age, gender, race, emotion, bboxes = face_analysis(image_filepath)
                prefix_results[pipeline_type][0].append(age)
                prefix_results[pipeline_type][1].append(gender)
                prefix_results[pipeline_type][2].append(race)
                prefix_results[pipeline_type][3].append(emotion)
                prefix_results[pipeline_type][4].append(bboxes)

                if pipeline_type not in image_filenames:
                    image_filenames[pipeline_type] = [image_filepath]
                else:
                    image_filenames[pipeline_type].append(image_filepath)

        # for x in range(0, 100):
        #     print(x+60)
        #     print(prefix_results["nothing"][3][x])
        #     print(prefix_results["fusion"][3][x])
        #     input("Next emotion")

    
        # Once we have all our prefixes, do some scoring.
        method_f1_scores = score_methods(prefix_results, image_filenames)
        # print(method_f1_scores)

        for method in method_f1_scores.keys():
            if method not in method_all_f1s.keys():
                method_all_f1s[method] = {}
                method_all_f1s[method]["age"] = [method_f1_scores[method]["age"]]
                method_all_f1s[method]["gender"] = [method_f1_scores[method]["gender"]]
                method_all_f1s[method]["race"] = [method_f1_scores[method]["race"]]
                method_all_f1s[method]["emotion"] = [method_f1_scores[method]["emotion"]]
            else: # This method already exists
                method_all_f1s[method]["age"].append(method_f1_scores[method]["age"])
                method_all_f1s[method]["gender"].append(method_f1_scores[method]["gender"])
                method_all_f1s[method]["race"].append(method_f1_scores[method]["race"])
                method_all_f1s[method]["emotion"].append(method_f1_scores[method]["emotion"])
        
    # Now, calculate average f1 for all metrics across all prefixes.
    print("\n ALL F1 Scores: \n")
    print(method_all_f1s)

    for method in method_all_f1s.keys():
        print("\nMETHOD:\n", method)
        print("Average Age F1: ", statistics.mean(method_all_f1s[method]["age"]))
        print("Average Gender F1: ", statistics.mean(method_all_f1s[method]["gender"]))
        print("Average Race F1: ", statistics.mean(method_all_f1s[method]["race"]))
        print("Average Emotion F1: ", statistics.mean(method_all_f1s[method]["emotion"]))

# input("Don't forget to use all prefixes, limit num images, AND change the limit on which methods")


# run_analysis()
