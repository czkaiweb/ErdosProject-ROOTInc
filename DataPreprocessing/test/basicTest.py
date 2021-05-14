from DataPreprocessing.python.Data import *


rootData = Data("/Users/czkaiweb/Research/ErdosBootCamp/Project/ProjectData/Root_Insurance_data.csv")

rootData.showFiles()
rootData.loadData()
rootData.splitData(fraction=[0.8,0,0.2],random_seed=12)

data = rootData.getDataCopy()

train_data = rootData.getTrainDataCopy()
print(train_data.head(5))

#set_of_combinations = set()
#for i in data.values:
#    set_of_combinations.add(tuple(i[:4]))
set_of_combinations = {('N', 1, 1, 'M'),
 ('N', 1, 1, 'S'),
 ('N', 1, 2, 'M'),
 ('N', 1, 2, 'S'),
 ('N', 2, 1, 'M'),
 ('N', 2, 1, 'S'),
 ('N', 2, 2, 'M'),
 ('N', 2, 2, 'S'),
 ('N', 3, 1, 'M'),
 ('N', 3, 1, 'S'),
 ('N', 3, 2, 'M'),
 ('N', 3, 2, 'S'),
 ('Y', 1, 1, 'M'),
 ('Y', 1, 1, 'S'),
 ('Y', 1, 2, 'M'),
 ('Y', 1, 2, 'S'),
 ('Y', 2, 1, 'M'),
 ('Y', 2, 1, 'S'),
 ('Y', 2, 2, 'M'),
 ('Y', 2, 2, 'S'),
 ('Y', 3, 1, 'M'),
 ('Y', 3, 1, 'S'),
 ('Y', 3, 2, 'M'),
 ('Y', 3, 2, 'S'),
 ('unknown', 1, 1, 'M'),
 ('unknown', 1, 1, 'S'),
 ('unknown', 1, 2, 'M'),
 ('unknown', 1, 2, 'S'),
 ('unknown', 2, 1, 'M'),
 ('unknown', 2, 1, 'S'),
 ('unknown', 2, 2, 'M'),
 ('unknown', 2, 2, 'S'),
 ('unknown', 3, 1, 'M'),
 ('unknown', 3, 1, 'S'),
 ('unknown', 3, 2, 'M')}


rootData.groupByFeature(featureSet=set_of_combinations, columns=["Currently Insured","Number of Vehicles","Number of Drivers","Marital Status"], name="GroupIndex")

print(rootData.Data_Train.head(5))