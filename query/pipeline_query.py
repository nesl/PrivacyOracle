import openai
import json
import os
import csv
import time

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


# Given some sensor data, we iterate through it.
def automated_mode(logfile, parent_folder, filename):
    messages = []

    # Open the file and get data
    message_options, message_states = open_file_and_get_data(parent_folder, filename)

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

    # Iterate through all prompts
    for p_i, prompt in enumerate(message_options[1:]):

        print("Current prompt: " + str(p_i) )
        messages.append(prompt)
        # chat = openai.ChatCompletion.create(
        #     model="gpt-4-1106-preview", messages=messages, seed=0, temperature=0.0
        # )
        chat = openai.ChatCompletion.create(
            model="gpt-4", messages=messages, seed=0, temperature=0.0
        )
        reply = chat.choices[0].message.content
        time.sleep(10)

        # Save the data to the logfile
        logfile.write("\n***sent***\n")
        logfile.write(prompt["content"])
        logfile.write("\n***reply***\n")
        logfile.write(reply)

        print(f"ChatGPT: {reply}")
        messages.append({"role": "assistant", "content": reply})

        # Clear previous messages
        messages = messages[:4]


if __name__ == "__main__":

    parent_folder = "chatgpt_prompts"
    filename = "query_privacy_pipeline"
    logfilepath = "results/" + filename + ".log"

    logfile = open(logfilepath, "w")

    # Index should be 1, 2, or 3
    automated_mode(logfile, parent_folder, filename)
    logfile.close()