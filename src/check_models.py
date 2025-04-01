#check_models.py

import os
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="This program checks if a model has already had its results taken (./results folder)")
    parser.add_argument("-m", help="The name of the model to check")
    parser.add_argument("-d", help="The name of the directory to check")
    
    args = parser.parse_args()

    for root, dirs, files in os.walk(args.d):
        for D in dirs:
            model_name = args.m.split('/')[-1]
            # print('----------------')
            # print(f"D: {D}")
            # print(f"model_name: {model_name}")
            if model_name == D:
                # print(f"model_name: {model_name}")    
                print("Model has already been trained")
                sys.exit(0)  # Exit with 0 (success) if found
    
    print("Model NOT yet trained")
    sys.exit(1)  # Exit with 1 (failure) if not found

if __name__ == "__main__":
    main()