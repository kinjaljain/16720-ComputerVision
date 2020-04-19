import os

with open('../python/data_bow/test/train_files.txt', 'r') as f:
    train = f.readlines()
test = '../python/data_bow/test/'
train = [test + x.rstrip() for x in train]

for file in train:
    os.remove(file)

with open('../python/data_bow/data_test/test_files.txt', 'r') as f:
    test = f.readlines()
data_test = '../python/data_bow/train/'
test = [data_test + x.rstrip() for x in test]

for file in test:
    os.remove(file)
