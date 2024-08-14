import os
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="This program checks if a model has already had its results taken (./results folder)")
    parser.add_argument("-r", help="The name of the model to check")
    parser.add_argument("-d", help="The name of the directory to check")
    parser.add_argument("--split", help="Split partition being checked")
    args = parser.parse_args()

    model_name = args.r.split('/')[-1]
    print('************* Results Checker ************')
    print(f'Model name being checked: {model_name}')
    print(f'Results dir: {args.d}')

    for root, dirs, files in os.walk(args.d):
        for F in files:
            if model_name in F:
                print(f"{model_name} has already been tested on the {args.split.upper()} split")
                print('**********************************')
                sys.exit(0)  # Exit with 0 (success) if found
    
    print("Model NOT yet tested")
    print('**********************************')
    sys.exit(1)  # Exit with 1 (failure) if not found

if __name__ == "__main__":
    main()