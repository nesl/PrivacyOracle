# Importing libraries 
import numpy as np 
import cv2 
import os

import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import sys


NUM_SKIPPED_IMAGES = 3


# Detect whether a face is present in the image.
def detect_face(image_path):


    image = cv2.cvtColor(cv2.imread(image_path), 
                        cv2.COLOR_BGR2RGB) 

    cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") 
    face_data = cascade.detectMultiScale(image, 
                                        scaleFactor=2.0, 
                                        minNeighbors=4)
    # face = face_analysis()
    # _, face_data, conf = face.face_detection(image_path=image_path, model='full')
    

    return face_data, image.shape, image

# Block entire image if a face is visible
def block_image(image_data):

    face_data, image_shape, image = image_data


    # Any kind of face data is detected
    if len(face_data):
        # Create a black image to omit info
        image = np.zeros(image_shape, dtype = np.uint8)

    return image
        

# blur faces
def blur_face(image_data):

    face_data, _, image = image_data

    for x, y, w, h in face_data: 
        # Draw a border around the detected face in Red 
        # colour with a thickness 5. 
        # image = cv2.rectangle(image, (x, y), (x+w, y+h),  
        #                     (255, 0, 0), 5) 
        image[y:y+h, x:x+w] = cv2.medianBlur(image[y:y+h, 
                                                x:x+w],  
                                            35) 
    
    # # Display the Blured Image 
    # print('Blurred Image') 
    # plt.imshow(image, cmap="gray") 
    # plt.axis('off') 
    # plt.show() 
    return image


# Load in an image
def load_image(image_path):
    image = cv2.cvtColor(cv2.imread(image_path), 
                        cv2.COLOR_BGR2RGB) 
    return image

# Save an image to a directory
def save_image_to_directory(image, out_folder):
    count = len(os.listdir(out_folder))
    cv2.imwrite(os.path.join(out_folder, "frame%d.jpg" % count), \
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

# Convert faces of one type to another
def facefusion(data):

    # Update our system path
    sys.path.append("./facefusion")
    from facefusion import core
    os.chdir("./facefusion")
    # For the face fusion, we just run their script
    core.cli()
    os.chdir("..")

    return 


# Let's convert a folder to a video
def convert_images_to_video(image_folder):

    outfile = "processed_video/temp1.mp4"

    images = [img for img in os.listdir(image_folder) if ".jpg" in img]
    images = sorted(images)
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    h,w,l = frame.shape

    video = cv2.VideoWriter(outfile, cv2.VideoWriter_fourcc(*'MP4V'), \
             30/NUM_SKIPPED_IMAGES,(w,h))

    # For every image, add it to our video
    for image in tqdm(images[::NUM_SKIPPED_IMAGES]):
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

    return outfile


# Now we can convert a video to a folder of images
def convert_video_to_images(image_folder):

    print("CONVERTING!!!")
    video_file = "processed_video/fusion.mp4"
    if not os.path.exists(image_folder):
        os.mkdir(image_folder)

    vidcap = cv2.VideoCapture(video_file)
    success,image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite(os.path.join(image_folder, "frame%d.jpg" % count), image)     # save frame as JPEG file      
        success,image = vidcap.read()
        count += 1

    return image_folder




# Run some experiments:
#  manifest is meant to be a list of functions that we execute in sequence.

# for blur and block
#  iterate through each folder
#  iterate through each image
    #  detect face
    #  block/blur face
    #  save image to folder

# for fusion
#  iterate through each folder
#  convert folder images into video
#  Execute facefusion on the video
#  Convert video back into a folder


def execute_module(module_name, curr_data, output_folder):

    out_data = []
    if module_name == "save_image_to_directory":
        save_image_to_directory(curr_data, output_folder)
    elif module_name == "convert_video_to_images":
        convert_video_to_images(output_folder)
    else:

        # Execute the module
        var_dict = {"curr_data": curr_data}
        program = 'out_data = ' + module_name + '(curr_data)'
        exec(program, globals(), var_dict)

        # Now, get the variables
        out_data = var_dict["out_data"]

    return out_data

        



def run_pipeline(img_folder, manifest, curr_data, output_folder):

    # Get the first data of the pipeline
    curr_data = img_folder
    # Check if we need an inner loop
    if "facefusion" in manifest:
        # Now, begin executing the pipeline
        for module in manifest:
            curr_data = execute_module(module, curr_data, output_folder)
    else: 
        # Iterate through each image
        images = [img for img in os.listdir(img_folder) if ".jpg" in img]
        images = sorted(images)

        for image_filepath in tqdm(images[::NUM_SKIPPED_IMAGES]):
            curr_data = os.path.join(img_folder, image_filepath)
            # curr_data = "./Chokepoint/P1E_S1/P1E_S1_C1/00000274.jpg"
            # Now, begin executing the pipeline
            for module in manifest:
                curr_data = execute_module(module, curr_data, output_folder)

def run_experiments(manifest, exp_name, img_folder):

    

    # List out some image folders
    # folders = ["./Chokepoint/P1E_S1/P1E_S1_C1"]

    # folders = ["./Chokepoint/P1E_S1/P1E_S1_C1",
    #            "./Chokepoint/P1E_S2/P1E_S2_C2",
    #            "./Chokepoint/P1E_S3/P1E_S3_C3"]

    output_parent_dir = "processed_video"

    if not os.path.exists(output_parent_dir):
        os.mkdir(output_parent_dir)
    
    

    # Get the output folder
    num_output_folders = len(os.listdir(output_parent_dir))

    # Iterate through each folder
    # for img_folder in folders:

    print("Executing manifest ", manifest, " for folder ", img_folder)

    out_folder = os.path.join( output_parent_dir, \
            img_folder.split("/")[-1] + "_" + str(num_output_folders))
    out_folder += "_" + exp_name

    if os.path.exists(out_folder):
        return

    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    

    # Now, execute the pipeline
    run_pipeline(img_folder, manifest, img_folder, out_folder)
    

# Parse the manifest string
def parse_manifest_str(manifest_str):

    manifest = manifest_str.split("[")[1].split("]")[0]
    manifest = [x.strip()[1:-1] for x in manifest.split(",")]
    return manifest

# Get the corresponding manifest for a operator based on the LLM responses
def get_manifests(llm_response_filepath):
    
    # Get the responses for the LLM
    response_file = open(llm_response_filepath, "r")
    llm_responses = response_file.read()
    response_file.close()

    # Split by replies
    llm_responses = llm_responses.split("***reply***")
    llm_responses = [x.split("***sent***")[0] for x in llm_responses]
    # Take only the last 3 responses
    llm_responses = llm_responses[-3:]

    # Now, parse each response
    block_manifest = parse_manifest_str(llm_responses[0])
    blur_manifest = parse_manifest_str(llm_responses[1])
    fusion_manifest = parse_manifest_str(llm_responses[2])
    
    return block_manifest, blur_manifest, fusion_manifest



def process_all_videos(dataset_path, llm_response_filepath):

    # Grab some of the data
    parent_folders = os.listdir(dataset_path)
    parent_folders = [x for x in parent_folders if "groundtruth" not in x]
    parent_folders = [x for x in parent_folders if "ignored" not in x]

    folders = []
    # Get the children
    for pfolder in parent_folders:
        pfolder_actual = os.path.join(dataset_path, pfolder)
        children_folders = os.listdir(os.path.join(pfolder_actual))
        children_folders = [x for x in children_folders if "tar.xz" not in x]
        children_folders = [os.path.join(pfolder_actual, x) for x in children_folders]
        folders.extend(children_folders)

    nothing_manifest = ["load_image", "save_image_to_directory"]
    block_manifest, blur_manifest, fusion_manifest = get_manifests(llm_response_filepath)
    


    # Now, iterate.
    for img_folder in folders:
        run_experiments(nothing_manifest, "nothing", img_folder)
        run_experiments(blur_manifest, "blur", img_folder)
        run_experiments(block_manifest, "block", img_folder)
        run_experiments(fusion_manifest, "fusion", img_folder)


