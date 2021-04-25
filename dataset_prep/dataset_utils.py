from itertools import compress
import os
import numpy as np

def clean_annotation(annotations, train_dir, test_dir):
  files = []
  for r, d, f in os.walk(train_dir):
    for file in f:
      #print(os.path.join(r, file))
      files.append(os.path.join(r, file))
  for r, d, f in os.walk(test_dir):
    for file in f:
      #print(os.path.join(r, file))
      files.append(os.path.join(r, file))

  mask = np.ones(len(annotations), dtype=bool)

  for i in range(len(annotations)):
    if f"{annotations[i][1]}" not in files:
      print(f"Deleting {annotations[i][1]}")
      mask[i] = False
  return list(compress(annotations, mask)), files

def _process_csv_file(file):
    with open(file, 'r') as fr:
        files = fr.readlines()
    return files

def generate_train_test_filenames():

    print("Generating Train-Test split")
    TRAINFILE = '/content/train_split_v3.txt'
    TESTFILE = '/content/test_split_v3.txt'
    dataset_train = _process_csv_file(TRAINFILE)
    dataset_test = _process_csv_file(TESTFILE)
    datasets = {'BENIGN': [], 'MALIGNANT': []}
    for l in dataset_train:
        entry = l.split()
        entry[1] = f"data/train/{entry[1]}"
        datasets[entry[2]].append(entry)
    for l in dataset_test:
        entry = l.split()
        entry[1] = f"data/test/{entry[1]}"
        datasets[entry[2]].append(entry)

    break_point_BENIGN = int(len(datasets['BENIGN'])/5)
    break_point_MALIGNANT = int(len(datasets['MALIGNANT'])/5)

    train_set = [datasets['BENIGN'][0:(break_point_BENIGN*4)] + datasets['MALIGNANT'][0:(break_point_MALIGNANT*4)]]
    test_set = [datasets['BENIGN'][(break_point_BENIGN*4):] + datasets['MALIGNANT'][(break_point_MALIGNANT*4):]]

    print("Cleaning train annotations...")
    annotations_train, filestr = clean_annotation(train_set[0], "data/train", "data/test")
    print("Cleaning test annotations...")
    annotations_test, fileste = clean_annotation(test_set[0], "data/train", "data/test")
    print('done!')

    return annotations_train, annotations_test