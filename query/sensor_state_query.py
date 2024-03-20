import openai
import json
import os
import csv
import time

# Function for reading from an ADL file
# Files are small enough to fit in memory, so just put all into same data structure
def read_data_adl(adl_filepath):

    adl_data = []  # List of (start time, end time, label)

    with open(adl_filepath, "r") as csvfile:
        filereader = csv.reader(csvfile, delimiter='\t')
        next(filereader) # Skip header rows
        next(filereader)
        for row in filereader:
            row = [x for x in row if len(x)]
            start_time = row[0]
            end_time = row[1]
            label = row[2]

            adl_data.append((start_time, end_time, label))
    
    return adl_data


# Function for reading from a sensor file
# Files are small enough to fit in memory, so just put all into same data structure
def read_data_sensor(sensor_filepath):

    sensor_data = {}  # Dict of date: [(start time, end time, sensor type, sensor placement) .. ]

    with open(sensor_filepath, "r") as csvfile:
        filereader = csv.reader(csvfile, delimiter='\t')
        next(filereader) # Skip header rows
        next(filereader)
        for row in filereader:

            row = [x for x in row if len(x)]
            # print(row)

            date = row[0].split()[0]

            if date not in sensor_data.keys():
                sensor_data[date] = []

            start_time = row[0].split()[1]
            end_time = row[1].split()[1]
            sensor_object = row[2]
            sensor_type = row[3]
            sensor_placement = row[4]

            # Get text format
            sensor_event = "From " + start_time + " to " + end_time + ", a " + \
                            sensor_type + " sensor on a " + sensor_object + " was activated." 

            # Get digital format
            sensor_event = (start_time, end_time, sensor_type, sensor_object)

            sensor_data[date].append(sensor_event)
    
    return sensor_data
        










# Load file of interest 
# This file basically includes a set of possible queries we can send to ChatGPT
#   where 0 is the preamble, and 1+ is one of the possible queries.

def open_file_and_get_data(parent_folder, filename):
    file_of_interest = parent_folder + "/" + filename
    with open(file_of_interest, "r") as f:

        message_prompts = json.load(f)

        message_options = message_prompts["data"]
        message_states = ""
        if "possibilities" in message_prompts:
            message_states = message_prompts["possibilities"]

    return message_options, message_states


# Create all the necessary prompts
def create_all_prompts(message_format, sensor_data):

    all_prompts = []
    days = []
    for day in sorted(sensor_data.keys()):
        
        # Get a current day's worth of sensor events
        current_sensor_events = '\n'.join([str(x) for x in sensor_data[day]])
        # Get the context prompt
        current_context_prompt = message_format["content"]

        # Now put the two together:
        current_content = current_context_prompt + "\n" + current_sensor_events

        entry = {"role": message_format["role"], "content": current_content}
        all_prompts.append(entry)
        days.append(day)
    
    return all_prompts, days

# Given some sensor data, we iterate through it.
def run_queries(sensor_data_file, logfile, message_option_index, message_options):

    # Get all the log data
    # read_data_adl("./UCI ADL Binary Dataset/OrdonezA_ADLs.txt")
    sdata = read_data_sensor(sensor_data_file)


    messages = []

    to_send = message_options[0]
    # Send initial prompt
    messages.append(to_send)
    # chat = openai.ChatCompletion.create(
    #     model="gpt-4-1106-preview", messages=messages, seed=0, temperature=0.0
    # )
    chat = openai.ChatCompletion.create(
        model="gpt-4", messages=messages, seed=0, temperature=0.0
    )
    reply = chat.choices[0].message.content

    # Save the data to the logfile
    logfile.write("\n***sent***\n")
    logfile.write(to_send["content"])
    logfile.write("\n***reply***\n")
    logfile.write(reply)

    print(f"ChatGPT: {reply}")
    messages.append({"role": "assistant", "content": reply})

    all_prompts, days = create_all_prompts(message_options[message_option_index], sdata)
    logfile.write("\n**days: " + str(days) + "\n")

    # Iterate through all prompts
    for p_i, prompt in enumerate(all_prompts):

        print("Current prompt: " + str(p_i) )
        messages.append(prompt)
        # chat = openai.ChatCompletion.create(
        #     model="gpt-4-1106-preview", messages=messages, seed=0, temperature=0.0
        # )
        chat = openai.ChatCompletion.create(
            model="gpt-4", messages=messages, seed=0, temperature=0.0
        )
        reply = chat.choices[0].message.content
        time.sleep(30)

        # Save the data to the logfile
        logfile.write("\n***sent***\n")
        logfile.write(prompt["content"])
        logfile.write("\n***reply***\n")
        logfile.write(reply)

        print(f"ChatGPT: {reply}")
        messages.append({"role": "assistant", "content": reply})

        # Clear previous messages
        messages = messages[:4]


def interactive_mode():

    messages = []

    # Loop until ctrl+c
    while True:

        # try:
        # Get the message
        #   This basically selects either a predefined message given by file_of_interest, or you can type 
        #   your own query.
        message = input("Select a message option (0-N), or type it here:")

        message_options, message_states = open_file_and_get_data()


        # Check if this is a predefined message or a custom response
        to_send = ""
        if message.isnumeric():
            to_send = message_options[int(message)]
        elif message == "quit":
            logfile.close()
            break
        else:
            to_send = {"role": "user", "content": message}

        #  Send the message to ChatGPT
        #  Note - we also send all historical messages between us and ChatGPT.
        if to_send:
            messages.append(to_send)
        
        # try:
        chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages
        )
        reply = chat.choices[0].message.content
        # except:
        #     reply = ""
        #     while len(reply) < 1:
                

        # Save the data to the logfile
        logfile.write("\n***sent***\n")
        logfile.write(to_send["content"])
        logfile.write("\n***reply***\n")
        logfile.write(reply)

        print(f"ChatGPT: {reply}")
        messages.append({"role": "assistant", "content": reply})

        # except:  # Close when things go wrong
        #     logfile.close()
        #     break


def automated_mode(parent_folder, filename): 


     # Loop through each file and index
    prompt_indexes = [1,2,3]
    files = ["datasets/UCI ADL Binary Dataset/OrdonezA_Sensors.txt",\
            "datasets/UCI ADL Binary Dataset/OrdonezB_Sensors.txt",]

    # Open the file and get data
    message_options, message_states = open_file_and_get_data(parent_folder,filename)

    for prompt_index in prompt_indexes:
        for datafile in files:

            # Open a log file
            logfilepath = "results/" + filename + "_" + str(prompt_index) + "_" + \
                datafile.split("/")[-1].split(".")[0] + ".log"

            # If this file exists already, we just continue
            if os.path.exists(logfilepath):
                continue

            logfile = open(logfilepath, "w")

            # Index should be 1, 2, or 3
            run_queries(datafile, logfile, prompt_index, message_options)
            logfile.close()


if __name__ == "__main__":


    parent_folder = "chatgpt_prompts"
    filename = "query_privacy_states"

    automated_mode(parent_folder, filename)