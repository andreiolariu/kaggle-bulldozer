import numpy as np
from pymongo import Connection, ASCENDING

import util

connection = Connection()
db = connection.bulldozer
auctions = db.train

auctions.drop()
auctions.ensure_index('tag')

datasets = {
    'test': util.get_test_df,
    'train': util.get_train_df,
}

# read machine index file
machines = {}
machine_df = util.get_machines_df()
for row in machine_df.iterrows():
    row = row[1]
    row = dict(row)
    row['YearMade'] = row.get('MfgYear')
    row.pop('MfgYear')
    machines[row['MachineID']] = row

# init some stuff
d = util.get_test_df()
convert = {'Undercarriage_Pad_Width': {'16 inch': 3, '15 inch': 2, '36 inch': 23, '31 inch': 18, '31.5 inch': 18.5, '32 inch': 19, '20 inch': 7, '34 inch': 21, '28 inch': 15, '22 inch': 9, '25 inch': 12, 'None or Unspecified': 0, '27 inch': 14, '24 inch': 11, '18 inch': 5, '14 inch': 1, '26 inch': 13, '30 inch': 17, '33 inch': 20}, 'Stick_Length': {'15\' 9"': 117, '13\' 10"': 94, '9\' 5"': 41, '11\' 10"': 70, '7\' 10"': 22, '9\' 7"': 43, '10\' 6"': 54, '6\' 3"': 3, 'None or Unspecified': 0, '9\' 6"': 42, '12\' 8"': 80, '10\' 10"': 58, '8\' 6"': 30, '13\' 9"': 93, '8\' 2"': 26, '19\' 8"': 164, '13\' 7"': 91, '9\' 2"': 38, '8\' 4"': 28, '15\' 4"': 112, '10\' 2"': 50, '12\' 10"': 82, '24\' 3"': 219, '11\' 0"': 60, '9\' 10"': 46, '8\' 10"': 34, '12\' 4"': 76, '9\' 8"': 44, '14\' 1"': 97}, 'Tire_Size': {'23.5"': 16.5, '13"': 6, 'None or Unspecified': 0, '15.5"': 8.5, '15.5': 8.5, '10"': 3, '10 inch': 3, '23.1"': 16.1, '23.5': 16.5, '14"': 7, '7.0"': 1, '17.5': 10.5, '29.5': 22.5, '20.5"': 13.5, '26.5': 19.5, '17.5"': 10.5, '20.5': 13.5}, 'ProductSize': {'Compact': 2, 'Mini': 0, 'Medium': 3, 'Large / Medium': 4, 'Large': 5, 'Small': 1}, 'Blade_Width': {'None or Unspecified': 0, "13'": 13, "12'": 12, "16'": 16, "14'": 14, "<12'": 10}, 'UsageBand': {'High': 2, 'Medium': 1, 'Low': 0}}
types = {'Motorgrader': {'index': 0, 'Unidentified': 0, '130.0 to 145.0 Horsepower': 2, '145.0 to 170.0 Horsepower': 3, '200.0 + Horsepower': 5, '45.0 to 130.0 Horsepower': 1, '170.0 to 200.0 Horsepower': 4}, 'Track Type Tractor, Dozer': {'index': 1, 'Unidentified': 0, '260.0 + Horsepower': 8, '75.0 to 85.0 Horsepower': 2, '160.0 to 190.0 Horsepower': 6, '20.0 to 75.0 Horsepower': 1, '105.0 to 130.0 Horsepower': 4, '190.0 to 260.0 Horsepower': 7, '85.0 to 105.0 Horsepower': 3, '130.0 to 160.0 Horsepower': 5}, 'Hydraulic Excavator, Track': {'50.0 to 66.0 Metric Tons': 19, '300.0 + Metric Tons': 23, '14.0 to 16.0 Metric Tons': 11, '3.0 to 4.0 Metric Tons': 4, '150.0 to 300.0 Metric Tons': 22, 'index': 2, '8.0 to 11.0 Metric Tons': 8, '2.0 to 3.0 Metric Tons': 3, '4.0 to 5.0 Metric Tons': 5, '24.0 to 28.0 Metric Tons': 15, '33.0 to 40.0 Metric Tons': 17, '16.0 to 19.0 Metric Tons': 12, '6.0 to 8.0 Metric Tons': 7, '28.0 to 33.0 Metric Tons': 16, 'Unidentified': 0, '12.0 to 14.0 Metric Tons': 10, '40.0 to 50.0 Metric Tons': 18, '0.0 to 2.0 Metric Tons': 2, '66.0 to 90.0 Metric Tons': 20, 'Unidentified (Compact Construction)': 1, '19.0 to 21.0 Metric Tons': 13, '4.0 to 6.0 Metric Tons': 5.5, '11.0 to 12.0 Metric Tons': 9, '21.0 to 24.0 Metric Tons': 14, '90.0 to 150.0 Metric Tons': 21, '5.0 to 6.0 Metric Tons': 6}, 'Wheel Loader': {'40.0 to 60.0 Horsepower': 2, '110.0 to 120.0 Horsepower': 7, 'Unidentified': 0, '225.0 to 250.0 Horsepower': 13, '200.0 to 225.0 Horsepower': 12, 'index': 3, '350.0 to 500.0 Horsepower': 16, '120.0 to 135.0 Horsepower': 8, '0.0 to 40.0 Horsepower': 1, '175.0 to 200.0 Horsepower': 11, '500.0 to 1000.0 Horsepower': 17, '80.0 to 90.0 Horsepower': 4, '275.0 to 350.0 Horsepower': 15, '100.0 to 110.0 Horsepower': 6, '90.0 to 100.0 Horsepower': 5, '60.0 to 80.0 Horsepower': 3, '250.0 to 275.0 Horsepower': 14, '150.0 to 175.0 Horsepower': 10, '135.0 to 150.0 Horsepower': 9, '1000.0 + Horsepower': 18}, 'Backhoe Loader': {'14.0 to 15.0 Ft Standard Digging Depth': 2, 'Unidentified': 0, 'index': 4, '16.0 + Ft Standard Digging Depth': 4, '0.0 to 14.0 Ft Standard Digging Depth': 1, '15.0 to 16.0 Ft Standard Digging Depth': 3}, 'Skid Steer Loader': {'976.0 to 1251.0 Lb Operating Capacity': 3, 'Unidentified': 0, '2701.0+ Lb Operating Capacity': 9, '0.0 to 701.0 Lb Operating Capacity': 1, '1251.0 to 1351.0 Lb Operating Capacity': 4, '1351.0 to 1601.0 Lb Operating Capacity': 5, '1601.0 to 1751.0 Lb Operating Capacity': 6, 'index': 5, '1751.0 to 2201.0 Lb Operating Capacity': 7, '2201.0 to 2701.0 Lb Operating Capacity': 8, '701.0 to 976.0 Lb Operating Capacity': 2}}
to_skip_mapping = ['Undercarriage_Pad_Width', 'Tire_Size', 'YearMade', 'UsageBand', 'Stick_Length', 'SalesID', 'SalePrice', 'saledate', 'ProductSize', 'ModelID', 'MachineID', 'MachineHoursCurrentMeter', 'Blade_Width', 'auctioneerID']
to_map = list(d.columns)
to_map.extend(['ProductGroupDesc', 'PrimarySizeBasis', 'PrimaryLower']) # not found in test file, taken from machine index
for key in to_skip_mapping:
    try:
        to_map.remove(key)
    except:
        print key
to_map = set(to_map)
values = {c: {} for c in to_map}

def remove_bad_values(row):
    # remove nans
    row = {key:value for key, value in row.iteritems() if not \
            (isinstance(value, float) and np.isnan(value))}
    
    # special rules
    for key, value in [('MachineHoursCurrentMeter', 0)]:
        if key in row and row[key] == value:
            row.pop(key)
    if 'YearMade' in row and not (1950 < row['YearMade'] < 2012):
        row.pop('YearMade')
    for key in ['Engine_Horsepower', 'PrimaryUpper', 'fiManufacturerDesc']:
        if key in row:
            row.pop(key)
    return row

for tag, function in datasets.iteritems():  
    
    dataset = function()
    l = util.Logger(len(dataset), 20000, tag='init')
    
    # convert to dictionaries, clean and add to mongo
    for row in dataset.iterrows():
        l.step()
        row = row[1]
        row = dict(row)
        
        row = remove_bad_values(row)
                
        # update from machine index file
        m_row = machines[row['MachineID']]
        for key in m_row:
            if key not in row:
                row[key] = machines[row['MachineID']][key]
        
        row = remove_bad_values(row)
                    
        # convert type
        t = row['fiProductClassDesc']
        t = t.split(' - ')
        # row['type1'] = types[t[0]]['index'] # same as ProductGroup
        row['type2'] = types[t[0]][t[1]] + types[t[0]]['index'] * 30
        
        # parse model info
        '''
        if 'fiBaseModel' in row:
            model = str(row['fiBaseModel'])
            row['model1'] = model
            model += str(row.get('fiSecondaryDesc', ''))
            row['model2'] = model
            model += str(row.get('fiModelSeries', ''))
            row['model3'] = model
            model += str(row.get('fiModelDescriptor', ''))
            row['model4'] = model
        '''
        
        # convert date
        date =  row['saledate']
        row.update({
            'sale_year': date.year,
            'sale_month': date.month,
            'sale_day': date.day,
        })
        
        key = 'auctioneerID'
        if key in row:
            row[key] = int(row[key])
            
        # convert certain fields
        for key in convert.keys():
            if key in row and row[key] is not None:
                row[key] = convert[key][row[key]]
        
        # map string values
        for k, v in row.items():
            if k in to_map:
                if v not in values[k]:
                    values[k][v] = len(values[k]) + 1
                row[k] = values[k][v]
        
        # put saleprice in log scale
        if 'SalePrice' in row:
            row['SalePrice'] = np.log(row['SalePrice'])
        
        if tag == 'train':
            if row['sale_year'] == 2011:
                row['tag'] = 'train2'
            elif row['sale_year'] == 2012:
                row['tag'] = 'train3'
            else:
                row['tag'] = 'train1'
        else:
            row['tag'] = tag
                
        auctions.insert([row])
