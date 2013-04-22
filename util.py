import time
import os
from dateutil.parser import parse
import simplejson as json

import pandas as pd
import numpy as np

f = open('settings.json', 'r')
SETTINGS = json.loads(f.read().replace('\n', ''))
f.close()

def get_paths():
    """
    Redefine data_path and submissions_path here to run the benchmarks on your machine
    """
    data_path = SETTINGS['data_path']
    submission_path = SETTINGS['submission_path']
    return data_path, submission_path

def get_train_df(data_path = None):
    if data_path is None:
        data_path, submission_path = get_paths()

    train = pd.read_csv(os.path.join(data_path, SETTINGS['train']),
        converters={"saledate": parse})
    return train 
    
def get_machines_df(data_path = None):
    if data_path is None:
        data_path, submission_path = get_paths()

    machines = pd.read_csv(os.path.join(data_path, SETTINGS['machines']))
    return machines 

def get_test_df(data_path = None):
    if data_path is None:
        data_path, submission_path = get_paths()

    test = pd.read_csv(os.path.join(data_path, SETTINGS['test']),
        converters={"saledate": parse})
    return test 

def get_train_test_df(data_path = None):
    return get_train_df(data_path), get_test_df(data_path)

def write_submission(submission_name, predictions, \
        IDs=None, submission_path=None):

    if submission_path is None:
        data_path, submission_path = get_paths()
    if IDs is None:
        test = get_test_df()    
        test = test.join(pd.DataFrame({"SalePrice": predictions}))
    else:
        test = pd.DataFrame({'SalesID': IDs, 'SalePrice': predictions})
        
    test[["SalesID", "SalePrice"]].to_csv(os.path.join(submission_path,
            submission_name), index=False)
        
def get_rmse(Y1, Y2):
    Y1 = np.array(Y1)
    Y2 = np.array(Y2)
    sq = (Y1 - Y2) ** 2
    return np.sqrt(np.mean(sq))
    
class Logger():
    def __init__(self, count, step, tag=None):
        self.count = count
        self.step_size = step
        self.time = time.time()
        self.tag = tag
    
    def step(self):
        self.count -= 1
        if self.count % self.step_size == 0:
            s = '%s - %s' % (self.count, time.time() - self.time)
            if self.tag:
                s = '[%s] %s' % (self.tag, s)
            print s
            self.time = time.time()
