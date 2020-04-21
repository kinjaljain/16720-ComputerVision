# import os
#
# with open('../python/data_bow/test/train_files.txt', 'r') as f:
#     train = f.readlines()
# test = '../python/data_bow/test/'
# train = [test + x.rstrip() for x in train]
#
# for file in train:
#     os.remove(file)
#
# with open('../python/data_bow/data_test/test_files.txt', 'r') as f:
#     test = f.readlines()
# data_test = '../python/data_bow/train/'
# test = [data_test + x.rstrip() for x in test]
#
# for file in test:
#     os.remove(file)

import os
with open("../data/train_files.txt", 'r') as f:
    train_file_names = f.readlines()
with open("../data/test_files.txt", 'r') as f:
    test_file_names = f.readlines()
dir_names = ["aquarium", "desert", "highway", "kitchen", "laundromat", "park", "waterfall", "windmill"]
for i in dir_names:
    os.makedirs("../data/train/" + i)
    os.makedirs("../data/test/" + i)
train_file_names = [x.strip() for x in train_file_names]
test_file_names = [x.strip() for x in test_file_names]
for i in train_file_names:
    os.system('cp ../data/' + i + ' ../data/train/' + i)
for i in test_file_names:
    os.system('cp ../data/' + i + ' ../data/test/' + i)
