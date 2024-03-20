import openai
import argparse
import os
from query.flow_validate_query import automated_mode as v_flow
from query.pipeline_query import automated_mode as p_flow
from query.sensor_state_query import automated_mode as s_flow

if __name__ == "__main__":

    
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", help="Flow validation, sensor state, or pipeline queries", default="flow")
    
    args = parser.parse_args()

    # Get our API key
    with open("API_KEY", "r") as f:
        openai.api_key = f.read().strip()

    # Now, check what type of query we are issuing:
    if args.exp == "flow":
        # Open a log file
        filename = "hipaa_3rdparty"
        logfilepath = "results/" + filename + "_" + str(len(os.listdir("results"))) + ".log"
        logfile = open(logfilepath, "w")
        # Run queries
        v_flow(logfile, "prompts", filename)
        logfile.close()

    elif args.exp == "pipeline":
        # Open a log file
        filename = "query_privacy_pipeline"
        logfilepath = "results/" + filename + "_" + str(len(os.listdir("results"))) + ".log"
        logfile = open(logfilepath, "w")
        # Run queries
        p_flow(logfile, "prompts", filename)
        logfile.close()

    elif args.exp == "state":
         # Open a log file
        filename = "query_privacy_states"
        # Run queries
        s_flow("prompts", filename)
    else:
        print("Invalid option - choose from 'flow', 'pipeline', or 'state'")

    
    