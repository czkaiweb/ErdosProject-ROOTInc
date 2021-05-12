from DataPreprocessing.python.Data import *


rootData = Data("/Users/czkaiweb/Research/ErdosBootCamp/Project/ProjectData/Root_Insurance_data.csv")

rootData.showFiles()
rootData.loadData()
rootData.splitData(fraction=[0.8,0,0.2],random_seed=12)

train_data = rootData.getTrainDataCopy()
print(train_data.head(5))

