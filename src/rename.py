import os

model_dir = './models'

for root, dirs, files in os.walk(model_dir):
    dirlist = sorted(os.listdir(root))
    for dir in dirlist:
        dir_path = os.path.join(root, dir)
        dir_path_new = f'{dir_path}_base'
        print(dir_path_new)
        os.rename(dir_path, dir_path_new)