import os
import argparse
import sys
import json

def main():
    parser = argparse.ArgumentParser(description="This program checks if a model has already had its results taken (./results folder)")
    parser.add_argument("-r", help="The name of the model to check")
    parser.add_argument("-d", help="The name of the directory to check")
    parser.add_argument("-n", help="Minimum number of test results per model", default=2)
    parser.add_argument("--split", help="Split partition being checked")
    args = parser.parse_args()

    model_name = args.r.split('/')[-1]
    print('************* Results Checker ************')
    print(f'Model results being checked: {os.path.join(args.d, model_name)}')
    print(f'Results dir: {args.d}')
    n_tests = 0
    for root, dirs, files in os.walk(args.d):
        for F in files:
            if model_name in F:
                json_filepath = os.path.join(root, F)
                with open(json_filepath, 'r', encoding='utf8') as f:
                    data = json.load(f)
                n_tests = len(data)
                if n_tests >= int(args.n):
                    print(f"{model_name} has already been tested {n_tests} times on the {args.split.upper()} split")
                    print('**********************************')
                    sys.exit(0)  # Exit with 0 (success) if found
    
    print(f"Model tested only {n_tests} times: testing...")
    print('**********************************')
    sys.exit(1)  # Exit with 1 (failure) if not found

if __name__ == "__main__":
    main()