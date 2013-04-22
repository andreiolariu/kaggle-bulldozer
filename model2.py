import time

import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.gaussian_process import GaussianProcess
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import NuSVR, SVR
import scipy

from util import Logger, get_rmse

class Linear():
    def __init__(self, type='Ridge', alpha=3, C=1.0, nu=0.2, limit=None, \
            epsilon=0.1):
        self.limit = limit
        if type == 'Ridge':
            self.model = Ridge(alpha=alpha)
        elif type == 'SVR':
            self.model = SVR(kernel='linear', C=C, epsilon=epsilon)
        elif type == 'NuSVR':
            self.model = NuSVR(C=C, nu=nu, kernel='linear')
        elif type == 'Lasso':
            self.model = Lasso(alpha=alpha)
        
    @staticmethod
    def get_cal(m):
        # get calitative features
        # watch out as indices depend on feature vector!
        return np.hstack((m[:,:23], m[:,24:37], m[:,38:52])) + 1
    
    @staticmethod
    def get_cant(m):
        # get cantitative features
        # watch out as indices depend on feature vector!
        return np.hstack((m[:,23:24], m[:,37:38], m[:,52:]))
        
    def fit(self, train_X, train_Y):
        # no fitting done here, just saving data
        if self.limit:
            if len(train_X) > self.limit:
                train_X = train_X[-self.limit:]
                train_Y = train_Y[-self.limit:]
        self.train_X = np.array(train_X)
        self.train_Y = np.array(train_Y)
        
        
    def predict(self, test_X):
        # fitting done here
        # not efficient on the long term
        test_X = np.array(test_X)
        enc = OneHotEncoder()
        scal = MinMaxScaler()
        data = np.vstack((self.train_X, test_X))
        enc.fit(self.get_cal(data))
        scal.fit(self.get_cant(data))
        
        new_train_X1 = enc.transform(self.get_cal(self.train_X))
        new_train_X2 = scal.transform(self.get_cant(self.train_X))
        new_train_X = scipy.sparse.hstack((new_train_X1, new_train_X2))
        new_test_X1 = enc.transform(self.get_cal(test_X))
        new_test_X2 = scal.transform(self.get_cant(test_X))
        new_test_X = scipy.sparse.hstack((new_test_X1, new_test_X2))
        
        self.model.fit(new_train_X, self.train_Y)
        R = self.model.predict(new_test_X)
        return R

def tt(args):
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, \
            ExtraTreesRegressor
    
    task_id, model_type, model_params, Xtrain, Ytrain, Xtest, model_key, key = args
    
    if len(Xtest) == 0:
        return (task_id, [])
        
    model_options = {
        'GBR': GradientBoostingRegressor,
        'RFR': RandomForestRegressor,
        'Linear': Linear,
        'ETR': ExtraTreesRegressor,
        
    }
    map_params = {
        'ne': 'n_estimators',
        'md': 'max_depth',
        'mss': 'min_samples_split',
        'msl': 'min_samples_leaf',
        'ss': 'subsample',
        'lr': 'learning_rate',
        'n': 'n_jobs',
        'rs': 'random_state',
        'a': 'alpha',
        't': 'type',
        'c': 'C',
        'nu': 'nu',
        'l': 'limit',
        'e': 'epsilon',
    }
    
    mp = {}
    for k, v in model_params.items():
        mp[map_params[k]] = v
    m = model_options[model_type](**mp)
    m.fit(Xtrain, Ytrain)
    r = m.predict(Xtest)
    del m
    del model_options
    return (task_id, r)

class SplitModel():
    '''
        build a series of models by spliting the dataset on a given feature
    '''
    def __init__(self, split_keys, index, n_est=5, models=None, \
                weights=None, bias=None, seed=1):
        self.bias = None
        
        model_params = []
        for md in [3, 6, 12, 16, 30]:
            for lr in [0.1, 0.3, 0.6, 0.9]:
                for mss in [5, 10, 16, 32, 64]:
                    msl = mss / 2
                    for ss in [0.3, 0.5, 0.7, 1]:
                        model_params.append(
                            ('GBR', dict(ne=n_est, md=md, lr=lr, mss=mss, msl=msl, ss=ss))
                        )
        for md in [3, 6, 12, 16, 30]:
            for mss in [5, 10, 16, 32, 64]:
                msl = mss / 2
                model_params.append(
                    ('RFR', dict(ne=n_est, md=md, mss=mss, msl=msl, n=4, rs=seed+md+mss))
                )
        model_params.append(('Linear', dict(t='NuSVR', c=0.1, nu=0.2, l=25000)))
        model_params.append(('Linear', dict(t='SVR', c=0.1, e=0.05, l=10000))) # ajuta pe index1 (0.22013316780439385, 0.52, 0.2, 0.14, 0.08, 0.06)
        model_params.append(('Linear', dict(t='Lasso', a=0.0001, l=20000)))
        model_params.append(('Linear', dict(t='Lasso', a=0.00001, l=20000)))
            
        self.bias = bias
        if weights:
            self.weights = weights
        else:
            self.weights = [1.0/len(models)] * len(models)
        self.model_params = []
        for i in range(len(models)):
            self.model_params.append((i,) + model_params[models[i]])
        for m in self.model_params:
            print m
            
        self.index = index
        self.split_keys = split_keys
        
    def train_test(self, Xtrain, Ytrain, Xtest):
        Xtrain = np.array(Xtrain)
        Ytrain = np.array(Ytrain)
        Xtest = np.array(Xtest)
        tasks = []
        task_id = 0
        results = []
        l = Logger(len(self.split_keys) * len(self.model_params), \
                len(self.split_keys), tag='SplitModel')
        for key in self.split_keys:
            mask_train = Xtrain[:,self.index] == key
            mask_test = Xtest[:,self.index] == key
            for model_key, model_type, model_params in self.model_params:
                task = (
                    task_id,
                    model_type, 
                    model_params, 
                    Xtrain[mask_train],
                    Ytrain[mask_train],
                    Xtest[mask_test],
                    model_key,
                    key,
                )
                results.append(tt(task))
                print (task_id, model_key, key)
                tasks.append((task_id, model_key, key))
                l.step()
                
                task_id += 1
        
        tasks = {t[0]: t[1:] for t in tasks}
        result_batches = [np.array([0.0] * len(Xtest))\
                        for i in range(len(self.model_params))]
        for result_set in results:
            task_id, Ytask = result_set
            task = tasks[task_id]
            model_key = task[-2]
            mask_test = Xtest[:,self.index] == task[-1]
            result_batches[model_key][mask_test] += Ytask
        
        Ytest = np.array([0.0] * len(Xtest))
        for (i, batch) in enumerate(result_batches):
            Ytest += batch * self.weights[i]
        if self.bias:
            Ytest += self.bias
        
        return (Ytest, result_batches)
        
        
class BigModel():
    def __init__(self, columns, n_est=5, seed=1):
        split_keys1 = sorted(list(range(1,7)))
        index1 = columns.index('ProductGroup')
        split_keys2 = [0, 1, 2, 3, 4, 5, 30, 31, 32, 33, 34, 35, 36, 37, 38, 60, 61, 62, 63, 64, 65, 65.5, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 120, 121, 122, 123, 124, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159]
        index2 = columns.index('type2')
        
        model_weights1 = [0.03131793, 0.02611096, 0.04683615, 0.04510745, 0.05901089, 0.05310125, 0.03905133, 0.06681227, 0.02479392, 0.04527772, 0.04649242, 0.03905532, 0.05324603, 0.02950949, 0.02942703, 0.04844027, 0.01096952, 0.04392747, 0.02677431, 0.00101947, 0.01306091, 0.00294178, 0.03421554, 0.05725074, 0.01947722, 0.01850144, -0.00323456, 0.03188587, 0.01445867, 0.01919743, 0.03913423, 0.02684197, -0.01231671, -0.01055348]
        model_indices1 = [20, 22, 26, 28, 30, 33, 36, 46, 57, 96, 97, 123, 131, 143, 160, 175, 185, 187, 197, 209, 215, 236, 255, 257, 276, 279, 298, 322, 330, 335, 354, 357, 414, 424]
        bias1 = -0.12037114510366997
        
        model_weights2 = [0.114999546, 0.0642159312, 0.0763160215, 0.0749201568, 0.0722169352, 0.0403322002, 0.105175222, 0.0257017482, 0.00992976551, -0.0198667402, 0.0836062323, 0.0618304965, -0.00770000674, -0.00243349526, 0.106124237, 0.0228227453, -1.57590333e-05, 0.0449772596, 0.0141671971, -0.0480243632, 0.049008765, 0.0389751147, 0.087701499]
        model_indices2 = [22, 24, 29, 47, 88, 94, 98, 110, 130, 139, 162, 172, 192, 214, 256, 291, 297, 322, 329, 371, 413, 414, 415]
        bias2 = -0.099892980166821133
        
        self.weights = [0.53, 0.2, 0.09, 0.1, 0.08]
        self.models = [
            SplitModel(split_keys1, index1, n_est=n_est, \
                seed=seed+1, models=model_indices1, weights=model_weights1,\
                bias=bias1),
            SplitModel(split_keys2, index2, n_est=n_est, \
                seed=seed+2, models=model_indices2, weights=model_weights2,\
                bias=bias2),
            SplitModel(split_keys1, index1, seed=seed+3, models=[425]),
            SplitModel(split_keys2, index2, seed=seed+4, models=[427]),
            SplitModel(split_keys2, index2, seed=seed+5, models=[428]),
        ]
    
    def train_test(self, train_X, train_Y, test_X):
        self.results = []
        rez = np.array([0.0] * len(test_X))
        l = Logger(len(self.models), 1, tag='BigModel')
        
        for i, m in enumerate(self.models):
            m_rez, m_batches = m.train_test(train_X, train_Y, test_X)
            rez += self.weights[i] * m_rez
            self.results.append((m_rez, m_batches))
            l.step()
        return rez

'''
class TimeModel():
    @staticmethod
    def train_test(train_X, train_Y, test_X):
        test_X_np = np.array(test_X)
        year1 = np.min(np.array(test_X_np)[:,37])
        month1 = np.min(np.array(test_X_np)[:,36])
        
        year2 = np.max(np.array(test_X_np)[:,37])
        month2 = np.max(np.array(test_X_np)[:,36])
        n_predict = int((year2 - year1) * 12 + month2 - month1 + 1)
        
        ppm = {} # price per month
        apm = [[] for i in range(13)] # average per month
        for i, row in enumerate(train_X):
            m = row[36]
            y = row[37]
            p = train_Y[i]
            
            if y * 12 + m < year1 * 12 + month1 - 72:
                continue
                
            if y not in ppm:
                ppm[y] = {}
            if m not in ppm[y]:
                ppm[y][m] = []
            ppm[y][m].append(p)
            apm[m].append(p)
            
        apm = [np.mean(l) for l in apm[1:]]
        average = np.mean(train_Y)
        
        plot = []
        for y in sorted(ppm.keys()):
            for m in sorted(ppm[y].keys()):
                plot.append(np.mean(ppm[y][m]))
        
        X = np.reshape(range(len(plot) + n_predict), (len(plot) + n_predict, 1))
        
        nuggets = [0.000003,0.00001,0.00003,0.0001,0.0003,0.001,0.003, 0.01]
        total_dev = np.array([0.0] * n_predict)
        preds = np.array([0.0] * len(X))
        for nugget in nuggets:
            g = GaussianProcess(regr='linear', \
                        corr='squared_exponential', nugget=nugget)
            g.fit(X[:-n_predict], plot)
            preds += g.predict(X)
            deviations = g.predict(X[-n_predict:]) - average
            total_dev = total_dev + deviations
        
        
        total_dev /= len(nuggets)
        preds /= len(nuggets)
        
        R = []
        for row in test_X:
            m = row[36] # month
            y = row[37] # year
            i = (y - year1) * 12 + m - month1
            R.append(total_dev[i])
        
        return R
'''
