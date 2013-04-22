import numpy as np
from pymongo import Connection

from model2 import BigModel
from util import Logger, write_submission, get_rmse

connection = Connection()
db = connection.bulldozer
auctions = db.train

columns = [
    'auctioneerID',
    'Backhoe_Mounting',
    'Blade_Extension',
    'Blade_Type',
    'Blade_Width',
    'Coupler_System',
    'Coupler',
    'datasource',
    'Differential_Type',
    'Drive_System',
    'Enclosure_Type',
    'Enclosure',
    'fiBaseModel',# replaced with modelX
    'fiManufacturerID', # in machine index
    'fiModelDesc',# replaced with modelX
    'fiModelDescriptor',# replaced with modelX
    'fiModelSeries',# replaced with modelX
    #'fiProductClassDesc', # redundant
    'fiSecondaryDesc',# replaced with modelX
    'Forks',
    'Grouser_Tracks',
    'Grouser_Type',
    'Hydraulics_Flow',
    'Hydraulics',
    'MachineHoursCurrentMeter',
    'MachineID',
    'ModelID',
    'Pad_Type',
    'Pattern_Changer',
    'PrimaryLower', # in machine index
    'PrimarySizeBasis', # in machine index
    'ProductGroup',
    'ProductSize',
    'Pushblock',
    'Ride_Control',
    'Ripper',
    'sale_day',
    'sale_month',
    'sale_year',
    #'saledate', # useless in random trees
    'Scarifier',
    'state',
    'Steering_Controls',
    'Stick_Length',
    'Stick',
    'Thumb',
    'Tip_Control',
    'Tire_Size',
    'Track_Type',
    'Transmission',
    'Travel_Controls',
    'type2',
    'Turbocharged',
    'Undercarriage_Pad_Width',
    'UsageBand',
    'YearMade',
]

Ycol = 'SalePrice'

def get_db_data(tags=['train1', 'train2']):
    records = auctions.find({'tag': {'$in': tags}})
    l = Logger(records.count(),20000, tag=str(tags))
    X = []
    Y = []
    IDs = []
    for record in records:
        l.step()        
        xr = [record.get(k, -1) for k in columns]# removed Nones
        X.append(xr)
        Y.append(record.get(Ycol))
        IDs.append(record['SalesID'])
    return (X, Y, IDs)

def run_test():
    
    #train_X, train_Y, _ = get_db_data(tags=['train1'])
    #test_X, test_Y, _ = get_db_data(tags=['train2'])
    train_X, train_Y, _ = get_db_data(tags=['train1', 'train2'])
    test_X, test_Y, _ = get_db_data(tags=['train3'])
    
    model = BigModel(columns, n_est=500)
    R2 = model.train_test(train_X, train_Y, test_X)
    print get_rmse(test_Y, R2)
    
    
'''
min_scor = 1
min_sol = None
C = 25
for i in range(C+1):
    for j in range(C+1-i):
        for k in range(C+1-i-j):
            for l in range(C+1-i-j-k):
                    m = C - i - j - k - l
                    i1 = i * 1.0 / C
                    j1 = j * 1.0 / C
                    k1 = k * 1.0 / C
                    l1 = l * 1.0 / C
                    m1 = m * 1.0 / C
                    scor = get_rmse(test_Y, r0*i1 + r1*j1 + r2*k1 + r3*l1 + ra*m1)
                    if scor < min_scor:
                        min_scor = scor
                        min_sol = (scor, i1,j1,k1, l1, m1)
                        print min_sol
'''                    
                    


def run_full():
    train_X, train_Y, _ = get_db_data(tags=['train1', 'train2', 'train3'])
    test_X, _, IDs = get_db_data(tags=['test'])
    
    m = BigModel(columns, n_est=300)
    R = m.train_test(train_X, train_Y, test_X)
    R = np.exp(R)
    write_submission("bigmodelv6.csv", R, IDs)

if __name__ == '__main__':
    run_full()
