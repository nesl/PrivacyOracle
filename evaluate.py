import argparse
import os
from evaluation.analysis_flows import analyze_results as f_analyze
from evaluation.analysis_sensor_states import analyze_results as s_analyze
from evaluation.analysis_pipeline import analyze_results as p_analyze
from evaluation.video_pipeline import process_all_videos

if __name__ == "__main__":

    
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", help="Flow validation, sensor state, or pipeline queries", default="flow")
    
    args = parser.parse_args()

    # Now, check what type of query we are issuing:
    if args.exp == "flow":
        #  Note - this is only meant to be used in conjunction with the 'user_study' prompt
        f_analyze("results/user_study_0.log")
    elif args.exp == "pipeline":
        process_all_videos("datasets/Chokepoint", "results/query_privacy_pipeline.log")
        p_analyze("processed_video")
    elif args.exp == "state":
        s_analyze("datasets/UCI ADL Binary Dataset/")
    else:
        print("Invalid option - choose from 'flow', 'pipeline', or 'state'")

    
    