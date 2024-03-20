import openai
import json
import os
import argparse

# Automatically generate the entries
#  when given something to permute over.
def create_all_prompts(message_format, message_options):

    all_prompts = []

    # Begin iterating through all message options
    for subject in message_options["subject"]:
        for sender in message_options["sender"]:
            for recipient in message_options["recipient"]:
                for datatype in message_options["datatype"]:
                    for principles in message_options["principles"]:

                        # Create the entry
                        content = message_format["content"].format(subject=subject,\
                             sender=sender, frecipient=recipient, fdatatype=datatype,\
                                 fprinciples=principles)
                        entry = {"role": message_format["role"], "content": content}
                        all_prompts.append(entry)


    print("There are " + str(len(all_prompts)) + " prompts")
    return all_prompts





# Load file of interest 
# This file basically includes a set of possible queries we can send to ChatGPT
#   where 0 is the preamble, and 1+ is one of the possible queries.

def open_file_and_get_data(prompt_folder, prompt_file):
    file_of_interest = prompt_folder + "/" + prompt_file
    with open(file_of_interest, "r") as f:

        message_prompts = json.load(f)

        message_options = message_prompts["data"]
        message_states = ""
        if "possibilities" in message_prompts:
            message_states = message_prompts["possibilities"]

    return message_options, message_states





def automated_mode(logfile, prompt_folder, prompt_file):

    messages = []

    # Open the file and get data
    message_options, message_states = open_file_and_get_data(prompt_folder, prompt_file)

    to_send = message_options[0]
    # Send initial prompt
    messages.append(to_send)
    chat = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=messages
    )
    reply = chat.choices[0].message.content

    # Save the data to the logfile
    logfile.write("\n***sent***\n")
    logfile.write(to_send["content"])
    logfile.write("\n***reply***\n")
    logfile.write(reply)

    print(f"ChatGPT: {reply}")
    messages.append({"role": "assistant", "content": reply})

    all_prompts = create_all_prompts(message_options[1], message_states)

    # Iterate through all prompts
    for p_i, prompt in enumerate(all_prompts):

        print("Current prompt: " + str(p_i) )
        messages.append(prompt)
        chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages
        )
        reply = chat.choices[0].message.content

        # Save the data to the logfile
        logfile.write("\n***sent***\n")
        logfile.write(prompt["content"])
        logfile.write("\n***reply***\n")
        logfile.write(reply)

        print(f"ChatGPT: {reply}")
        messages.append({"role": "assistant", "content": reply})

        # Clear previous messages
        messages = messages[:6]


def interactive_mode():

    messages = []

    # Loop until ctrl+c
    while True:

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
        chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages
        )
        reply = chat.choices[0].message.content

        # Save the data to the logfile
        logfile.write("\n***sent***\n")
        logfile.write(to_send["content"])
        logfile.write("\n***reply***\n")
        logfile.write(reply)

        print(f"ChatGPT: {reply}")
        messages.append({"role": "assistant", "content": reply})





if __name__ == "__main__":

    
    parser = argparse.ArgumentParser()
    parser.add_argument("--pfolder", help="Folder path to prompts", default="chatgpt_prompts")
    parser.add_argument("--pfile", help="Specific prompt file of interest", default="hipaa_3rdparty")
    args = parser.parse_args()

    # Get our API key
    with open("API_KEY", "r") as f:
        openai.api_key = f.read().strip()

    # Open a log file
    logfilepath = "results/" + args.pfile + "_" + str(len(os.listdir("results"))) + ".log"
    logfile = open(logfilepath, "w")

    automated_mode(logfile, args.pfolder, args.pfile)