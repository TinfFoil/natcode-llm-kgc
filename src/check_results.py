import os
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="This program checks if a model has already had its results taken (./results folder)")
    parser.add_argument("-m", help="The name of the model to check")
    parser.add_argument("-d", help="The name of the directory to check")
    
    args = parser.parse_args()

    for root, dirs, files in os.walk(args.d):
        for F in files:
            if args.m.split('/')[-1] in F:
                print("Model has already been tested")
                sys.exit(0)  # Exit with 0 (success) if found
    
    print("Model NOT yet tested")
    sys.exit(1)  # Exit with 1 (failure) if not found

if __name__ == "__main__":
    main()