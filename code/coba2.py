from openpyxl import load_workbook
import function as fc
import time
import numpy as np

start_time = time.time()
#matriksTarget = [[1,0,0,0,0,0,0],
#                 [0,1,0,0,0,0,0],
#                 [0,0,1,0,0,0,0],
#                 [0,0,0,1,0,0,0],
#                 [0,0,0,0,1,0,0],
#                 [0,0,0,0,0,1,0],
#                 [0,0,0,0,0,0,1]]
matriksTarget = [[0,0,1],
                 [0,1,0],
                 [0,1,1],
                 [1,0,0],
                 [1,0,1],
                 [1,1,0],
                 [1,1,1]]
np.savetxt('matriksTarget.txt',matriksTarget)
print 'Open File...'
wb = load_workbook('data1.xlsx')
sheet = wb.active
dataSet = fc.getData(sheet)
dataPre = fc.preprocessing(dataSet)
print 'Counting document for each class...'
docEachClass = fc.docEachClass(matriksTarget,dataSet,dataPre)
print 'Counting document for each class given word...'
Fw = fc.docGivenWord(matriksTarget,dataPre,dataSet)
print 'Counting Probability for Each Class Given Word w...'
piW = fc.probEachClassGivenWord(Fw,docEachClass,matriksTarget)
print 'Counting Information Gain for Each Class and Word...'
igW = fc.informationGain(docEachClass,dataSet,piW,dataPre)

#get words if igW value > threshold
ig = [x for x in igW if x[1] > 0.1]
igWThreshold = []
for i in ig:
    igWThreshold.append(i[0])
np.savetxt("igWThreshold.txt", igWThreshold, delimiter=",", fmt="%s")
print 'Counting TF...'
tf = fc.tf(igWThreshold,dataSet)
print 'Counting IDF...'
idf = fc.idf(tf,dataSet)
print 'Counting TFxIDF...'
tfIDF = fc.tfIDF(tf,idf,dataSet)

#start training ann (artificial neural network)
input_p = len(tfIDF[0][1])
hidden_p = 10
output_p = 3
lr = 0.07
epoch = 1000
mseStandar = 0.01
i_param = [input_p,hidden_p,output_p,lr,epoch,mseStandar]
np.savetxt('inputParameters.txt',i_param)
W1,W2,B1,B2 = fc.training(input_p,hidden_p,output_p,lr,epoch,mseStandar,tfIDF,dataSet)

##start testing
#print 'Now is testing the data...'
#wbTest = load_workbook('data.xlsx')
#sheetTest = wbTest.active
#dataSetTest = fc.getData(sheetTest)
#print 'Counting TF...'
#tfTest = fc.tf(igWThreshold,dataSetTest)
#print 'Counting IDF...'
#idfTest = fc.idf(tfTest,dataSetTest)
#print 'Counting TFxIDF...'
#tfIDFTest = fc.tfIDF(tfTest,idfTest,dataSetTest)
#label = fc.testing(tfIDFTest,W1,W2,B1,B2)
#hLoss = fc.hammingLoss(label,dataSetTest)
#print 'Hamming Loss = ', hLoss

print time.time() - start_time , ' seconds'