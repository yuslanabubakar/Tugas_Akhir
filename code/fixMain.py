# -*- coding: utf-8 -*-
"""
Created on Wed May 09 11:40:38 2018

@author: M. Yuslan Abubakar
"""

from openpyxl import load_workbook
import fixFunction as fc
import time
from collections import namedtuple
import numpy as np

start_time = time.time()
#matriksTarget = [[1,0,0,0,0,0,0],
#                 [0,1,0,0,0,0,0],
#                 [0,0,1,0,0,0,0],
#                 [0,0,0,1,0,0,0],
#                 [0,0,0,0,1,0,0],
#                 [0,0,0,0,0,1,0],
#                 [0,0,0,0,0,0,1]]
#matriksTarget = [[0,0,1],
#                 [0,1,0],
#                 [0,1,1],
#                 [1,0,0],
#                 [1,0,1],
#                 [1,1,0],
#                 [1,1,1]]
#np.savetxt('matriksTarget.txt',matriksTarget)
print 'Open File...'
wb = load_workbook('data1.xlsx')
sheet = wb.active
data = fc.getData(sheet)
print 'Now Start Remove Stopwords...'
preprocessing = fc.preprocessing(data)
##print 'Counting document for each class...'
##docEachClass = fc.docEachClass(dataAnjur,dataLarang,dataInfo)
print 'Counting document for each class given word...'
FW = fc.docGivenWord(data,preprocessing)
print 'Counting Probability for Each Class Given Word w...'
PIW = fc.probEachClassGivenWord(FW,data)
print 'Counting Information Gain for Each Class and Word...'
IGW = fc.informationGain(PIW)
#
#get words if igW value > threshold
#igwAnjur = [x for x in igAnjur if x[1] >= 0.05]
#igwLarang = [x for x in igLarang if x[1] >= 0.05]
#igwInfo = [x for x in igInfo if x[1] >= 0.05]
#igWThresholdAnjur = [x[0] for x in igwAnjur]
#igWThresholdLarang = [x[0] for x in igwLarang]
#igWThresholdInfo = [x[0] for x in igwInfo]
igw = []
for i in range(len(IGW)):
    igw.append([x for x in IGW[i] if x[1] >= 0.05])
    
for i in range(len(igw)):
    igw[i] = [x[0] for x in igw[i]]

IGW = namedtuple('IGW','anjuran, larangan, informasi')
IGW(igw[0],igw[1],igw[2])

##for i in ig:
##    igWThreshold.append(i[0])
##np.savetxt("igWThreshold.txt", igWThreshold, delimiter=",", fmt="%s")
#print 'Counting TF...'
#tfAnjur,tfLarang,tfInfo = fc.tf(igWThresholdAnjur,igWThresholdLarang,igWThresholdInfo,dataAnjur,dataLarang,dataInfo)
#print 'Counting IDF...'
#idfAnjur,idfLarang,idfInfo = fc.idf(tfAnjur,tfLarang,tfInfo,dataAnjur,dataLarang,dataInfo)
#print 'Counting TFxIDF...'
#tfidfAnjur,tfidfLarang,tfidfInfo = fc.tfIDF(tfAnjur,tfLarang,tfInfo,dataAnjur,dataLarang,dataInfo,idfAnjur,idfLarang,idfInfo)
##
##start training ann (artificial neural network)
#input_p_anjur = len(tfidfAnjur[0][1])
#input_p_larang = len(tfidfLarang[0][1])
#input_p_info = len(tfidfInfo[0][1])
#hidden_p = 10
#output_p = 1
#lr = 0.07
#epoch = 1000
#mseStandar = 0.001
##i_param = [input_p,hidden_p,output_p,lr,epoch,mseStandar]
##np.savetxt('inputParameters.txt',i_param)
#print 'Start training data...'
#W1Anjur,W2Anjur,B1Anjur,B2Anjur = fc.trainingAnjur(input_p_anjur,hidden_p,output_p,lr,epoch,mseStandar,tfidfAnjur)
#W1Larang,W2Larang,B1Larang,B2Larang = fc.trainingLarang(input_p_larang,hidden_p,output_p,lr,epoch,mseStandar,tfidfLarang)
#W1Info,W2Info,B1Info,B2Info = fc.trainingInfo(input_p_info,hidden_p,output_p,lr,epoch,mseStandar,tfidfInfo)
#print 'Finish'
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