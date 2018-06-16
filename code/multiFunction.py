# -*- coding: utf-8 -*-
"""
Created on Wed May 09 11:11:02 2018

@author: M. Yuslan Abubakar
"""

import re
import numpy as np
import math
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from collections import namedtuple
from operator import itemgetter

# create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def getData(sheet):
    dataSet = []
    for i in range(1, sheet.max_row+1):
        dataText = []
        label = []
        data = []
        dataText.append(sheet.cell(row=i, column=1).value)
        label.append(int(sheet.cell(row=i, column=2).value))
        label.append(int(sheet.cell(row=i, column=3).value))
        label.append(int(sheet.cell(row=i, column=4).value))
        data.append(dataText)
        data.append(label)
        dataSet.append(data)

    return dataSet

def preprocessing(dataSet):
    data = []
    for i in dataSet:
        data.append(i[0])
    f = open('stopwords.txt','r')
    stopwords = f.read()

    def stop_words(data):
        textStop = []
        for i in data:
            for x in i[0].lower().split():
                x = re.sub('[(;:,.\'`?!0123456789)]', '', x)
#                x = stemmer.stem(x.encode('utf-8')) #stemming
                if x.encode('utf-8') not in stopwords:
                    textStop.append(x.encode('utf-8'))
        return textStop

    dataStop  = stop_words(data)
    dataStop = np.unique(dataStop)
    thefile = open('dataStop.txt','w')
    for item in dataStop:
        print>>thefile, item
    print 'Preprocessing file saved'
    return dataStop  

def docEachClass(label,dataSet):
    data = []
    thefile = open('docEachClass.txt','w')
    for i in range(len(label)):
        d = []
        count = 0
        for x in dataSet:
            if x[1][i] == 1:
                count = count + 1
        d.append(label[i])
        d.append(count)
        data.append(d)
        print>>thefile, d
    
    return data

def docGivenWord(dataSet,preprocessing,label):
    result = []
    for i in preprocessing:
        for j in range(len(label)):
            data = []
            countDoc = 0
            for k in dataSet:
                if k[1][j] == 1:
                    c = 0
                    words = []
                    for word in k[0][0].lower().split():
                        word = re.sub('[(;:,.\'`?!0123456789)]', '', word)
#                        word = stemmer.stem(word.encode('utf-8')) #stemming
                        words.append(word.encode('utf-8'))
                    c = list(words).count(i)
                    if c > 0:
                        countDoc = countDoc + 1
            data.append(i)
            data.append(label[j])
            data.append(countDoc)
            result.append(data)
    
    thefile = open('Fw.txt','w')
    for item in result:
        print>>thefile, item
        
    return result

def probEachClassGivenWord(FW, label, docEachClass):
    result = []
    for i in FW:
        for j in docEachClass:
            a = []
            if i[1] == j[0]:
                piW = (i[2]+1) / float(j[1]+1) #smoothing
                a.append(i[0])
                a.append(i[1])
                a.append(piW)
                result.append(a)
                
    return result

def informationGain(preprocessing,PIW,label,docEachClass,dataSet):
    result = []
    thefile = open('informationGain.txt','w')
    for word in preprocessing:
        nilai = 0
        data = []
        for i in PIW:
            if i[0] == word:
                for j in docEachClass:
                    if i[1] == j[0]:
                        nilai = nilai + ((j[1] / float(len(dataSet))) * (-(i[2] * math.log(i[2],2))))
        nilai = 1 - nilai
        data.append(word)
        data.append(nilai)
        result.append(data)
        print>>thefile,data
    return result

def plotIGW(igW):
    x = []
    y = []
    for i in igW:
        x.append(i[0])
        y.append(i[1])
    
    y_pos = np.arange(len(x))
    
    plt.bar(y_pos, y, align='center', alpha=0.5)
    plt.xticks(y_pos, x)
    plt.ylabel('IG')
    plt.title('Information Gain')
    
    plt.show()

def thresholdIGW(IGW,value):
    return [x[0] for x in IGW if x[1] >= value]
    
def tf(IGW,dataSet):
    document = []
    for i in dataSet:
        document.append(i[0])
    TF = []
    for word in IGW:
        hasil = []
        l = []
        doc = 1
        for j in document:
            d = []
            words = []
            for x in j[0].lower().split():
                x = re.sub('[(;:,.\'`?!0123456789)]', '', x)
#                x = stemmer.stem(x.encode('utf-8')) #stemming
                words.append(x.encode('utf-8'))
            c = list(words).count(word)
            d.append(doc)
            d.append(c)
            hasil.append(d)
            doc = doc + 1
        l.append(word)
        l.append(hasil)
        TF.append(l)
    return TF

def idf(TF,dataSet):
    IDF = []
    for i in TF:
        d = []
        counts = 0
        for j in i[1]:
            if j[1] > 0:
                counts = counts + 1
        counts = math.log((len(dataSet)+len(TF))/float(counts+1)) #smoothing
        d.append(j[0])
        d.append(counts)
        IDF.append(d)
    return IDF

def tfIDF(TF,dataSet,IDF):
    TFIDF = []
    for j in range(len(TF)):
        tfidf = 0
        d = []
        l = []
        for k in TF[j][1]:
            dt = []
            tfidf = k[1] * IDF[j][1]
            dt.append(k[0])
            dt.append(tfidf)
            d.append(dt)
        l.append(TF[j][0])
        l.append(d)
        TFIDF.append(l)
    tfidf = []
    for j in range(len(dataSet)):
        d = []
        li = []
        for k in TFIDF:
            dt = []
            for l in k[1]:
                if j+1 == l[0]:
                    dt.append(k[0])
                    dt.append(l[1])
            d.append(dt)
        li.append(j+1)
        li.append(d)
        tfidf.append(li)
    
    return tfidf

def trainingAnjur(input_p,hidden_p,output_p,lr,epoch,mseStandar,tdidf,dataSet):
    X = []
    for i in tdidf:
        data = []
        for j in i[1]:
            data.append(j[1])
        X.append(data)
    
    #start training
    mse = 999999999
    iterate = 0
    np.random.seed(1)
    B1Anjur = np.matrix(np.random.uniform(0,1,hidden_p))
    np.random.seed(1)
    B2Anjur = np.matrix(np.random.uniform(0,1,output_p))
    np.random.seed(1)
    W1Anjur = np.random.uniform(0,1,(input_p,hidden_p))
    np.random.seed(1)
    W2Anjur = np.random.uniform(0,1,(hidden_p,output_p))
    while (mse > mseStandar and iterate < epoch):
        error = []
        mse = 0
        for i in range(len(X)):
            #feedforward
            maxi = max(X[i])
            mini = min(X[i])
            pp = [((x-mini)/(maxi-mini)) if (maxi-mini)>0 else 0 for x in X[i]]
            p = np.matrix(pp)
            V1 = np.dot(p,W1Anjur)
            V1 = V1 + B1Anjur
            A1 = 1 / (1 + np.exp(-0.01*V1))
            V2 = np.dot(A1,W2Anjur)
            V2 = V2 + B2Anjur
            A2 = 1 / (1 + np.exp(-0.01 * V2))
            e = (dataSet[i][1][0] - A2)
            error.append(e)
            
            #backpropagation
            D2 = np.multiply(np.multiply(A2,(1-A2)),error[i])
            dw2 = lr * np.dot(np.transpose(D2),A1)
            dw2 = np.transpose(dw2)
            D1 = np.multiply(A1,(1-A1))
            D1 = np.multiply(D1,np.transpose(np.dot(W2Anjur,np.transpose(D2))))
            dw1 = lr * np.dot(np.transpose(D1),p)
            dw1 = np.transpose(dw1)
            db1 = lr * D1
            db2 = lr * D2
            W1Anjur = dw1 + W1Anjur
            W2Anjur = dw2 + W2Anjur
            B1Anjur = db1 + B1Anjur
            B2Anjur = db2 + B2Anjur
        for e in error:
            mse += e**2
        mse = mse / len(error)
        iterate += 1
    print 'MSE Anjuran = ',mse
    print 'Epoch Anjuran = ',iterate
#    np.savetxt('W1.txt',W1)
#    np.savetxt('W2.txt',W2)
#    np.savetxt('B1.txt',B1)
#    np.savetxt('B2.txt',B2)
    return W1Anjur,W2Anjur,B1Anjur,B2Anjur
    
def trainingLarang(input_p,hidden_p,output_p,lr,epoch,mseStandar,tfidf,dataSet):
    X = []
    for i in tfidf:
        data = []
        for j in i[1]:
            data.append(j[1])
        X.append(data)
    
    #start training
    mse = 999999999
    iterate = 0
    np.random.seed(1)
    B1Larang = np.matrix(np.random.uniform(0,1,hidden_p))
    np.random.seed(1)
    B2Larang = np.matrix(np.random.uniform(0,1,output_p))
    np.random.seed(1)
    W1Larang = np.random.uniform(0,1,(input_p,hidden_p))
    np.random.seed(1)
    W2Larang = np.random.uniform(0,1,(hidden_p,output_p))
    while (mse > mseStandar and iterate < epoch):
        error = []
        mse = 0
        for i in range(len(X)):
            #feedforward
            maxi = max(X[i])
            mini = min(X[i])
            pp = [((x-mini)/(maxi-mini)) if (maxi-mini)>0 else 0 for x in X[i]]
            p = np.matrix(pp)
            V1 = np.dot(p,W1Larang)
            V1 = V1 + B1Larang
            A1 = 1 / (1 + np.exp(-0.01*V1))
            V2 = np.dot(A1,W2Larang)
            V2 = V2 + B2Larang
            A2 = 1 / (1 + np.exp(-0.01 * V2))
            e = (dataSet[i][1][1] - A2)
            error.append(e)
            
            #backpropagation
            D2 = np.multiply(np.multiply(A2,(1-A2)),error[i])
            dw2 = lr * np.dot(np.transpose(D2),A1)
            dw2 = np.transpose(dw2)
            D1 = np.multiply(A1,(1-A1))
            D1 = np.multiply(D1,np.transpose(np.dot(W2Larang,np.transpose(D2))))
            dw1 = lr * np.dot(np.transpose(D1),p)
            dw1 = np.transpose(dw1)
            db1 = lr * D1
            db2 = lr * D2
            W1Larang = dw1 + W1Larang
            W2Larang = dw2 + W2Larang
            B1Larang = db1 + B1Larang
            B2Larang = db2 + B2Larang
        for e in error:
            mse += e**2
        mse = mse / len(error)
        iterate += 1
    print 'MSE Larangan = ',mse
    print 'Epoch Larangan = ',iterate
#    np.savetxt('W1.txt',W1)
#    np.savetxt('W2.txt',W2)
#    np.savetxt('B1.txt',B1)
#    np.savetxt('B2.txt',B2)
    return W1Larang,W2Larang,B1Larang,B2Larang

def trainingInfo(input_p,hidden_p,output_p,lr,epoch,mseStandar,tfidf,dataSet):
    X = []
    for i in tfidf:
        data = []
        for j in i[1]:
            data.append(j[1])
        X.append(data)
    
    #start training
    mse = 999999999
    iterate = 0
    np.random.seed(1)
    B1Info = np.matrix(np.random.uniform(0,1,hidden_p))
    np.random.seed(1)
    B2Info = np.matrix(np.random.uniform(0,1,output_p))
    np.random.seed(1)
    W1Info = np.random.uniform(0,1,(input_p,hidden_p))
    np.random.seed(1)
    W2Info = np.random.uniform(0,1,(hidden_p,output_p))
    while (mse > mseStandar and iterate < epoch):
        error = []
        mse = 0
        for i in range(len(X)):
            #feedforward
            maxi = max(X[i])
            mini = min(X[i])
            pp = [((x-mini)/(maxi-mini)) if (maxi-mini)>0 else 0 for x in X[i]]
            p = np.matrix(pp)
            V1 = np.dot(p,W1Info)
            V1 = V1 + B1Info
            A1 = 1 / (1 + np.exp(-0.01*V1))
            V2 = np.dot(A1,W2Info)
            V2 = V2 + B2Info
            A2 = 1 / (1 + np.exp(-0.01 * V2))
            e = (dataSet[i][1][2] - A2)
            error.append(e)
            
            #backpropagation
            D2 = np.multiply(np.multiply(A2,(1-A2)),error[i])
            dw2 = lr * np.dot(np.transpose(D2),A1)
            dw2 = np.transpose(dw2)
            D1 = np.multiply(A1,(1-A1))
            D1 = np.multiply(D1,np.transpose(np.dot(W2Info,np.transpose(D2))))
            dw1 = lr * np.dot(np.transpose(D1),p)
            dw1 = np.transpose(dw1)
            db1 = lr * D1
            db2 = lr * D2
            W1Info = dw1 + W1Info
            W2Info = dw2 + W2Info
            B1Info = db1 + B1Info
            B2Info = db2 + B2Info
        for e in error:
            mse += e**2
        mse = mse / len(error)
        iterate += 1
    print 'MSE Informasi = ',mse
    print 'Epoch Informasi = ',iterate
#    np.savetxt('W1.txt',W1)
#    np.savetxt('W2.txt',W2)
#    np.savetxt('B1.txt',B1)
#    np.savetxt('B2.txt',B2)
    return W1Info,W2Info,B1Info,B2Info

def testing(tfIDFTest,W1,W2,B1,B2):
    X = []
    for i in tfIDFTest:
        data = []
        for j in i[1]:
            data.append(j[1])
        X.append(data)
    
    #start testing
    print 'Now start testing data...'
    label = []
    for i in range(len(X)):
        #feedforward
#        data = []
        maxi = max(X[i])
        mini = min(X[i])
        pp = [((x-mini)/(maxi-mini)) if (maxi-mini)>0 else 0 for x in X[i]]
        p = np.matrix(pp)
#        for j in range(len(p)):
#            p[j] = xmin + (((p[j] - min(p))*(xmax - xmin)) / (max(p) - min(p)))
        V1 = np.dot(p,W1)
        V1 = V1 + B1
        A1 = 1 / (1 + np.exp(-0.01*V1))
        V2 = np.dot(A1,W2)
        V2 = V2 + B2
        A2 = 1 / (1 + np.exp(-0.01 * V2))
        if A2 >= 0.5:
            label.append(1)
        else:
            label.append(0)
    print 'Finish'
    return label
        
def hammingLoss(labelAnjur,labelLarang,labelInfo,dataSetTest,label):
    target = []
    for i in dataSetTest:
        target.append(i[1])
    
    errorH = 0
    for i in range(len(labelAnjur)):
#        if (labelAnjur[i] != target[i][0]) or (labelLarang[i] != target[i][1]) or (labelInfo[i] != target[i][2]):
#            errorH += 1
        if (labelAnjur[i] != target[i][0]):
            errorH += 1
        if (labelLarang[i] != target[i][1]):
            errorH += 1
        if (labelInfo[i] != target[i][2]):
            errorH += 1
    
#    labelClass = len([list(x) for x in set(tuple(x) for x in target)])
    hLoss = (1/float(len(label))) * (1/float(len(dataSetTest))) * errorH
    return hLoss

def getWeights():
    W1 = np.matrix(np.loadtxt('W1.txt'))
    W2 = np.matrix(np.loadtxt('W2.txt'))
    B1 = np.matrix(np.loadtxt('B1.txt'))
    B2 = np.matrix(np.loadtxt('B2.txt'))
    return W1,W2,B1,B2

def getMatriksTarget():
    return np.loadtxt('matriksTarget.txt').tolist()

def getIGWThreshold():
    with open('igWThreshold.txt') as f:
        alist = f.read().splitlines(True)
    return alist

def getInputParameters():
    i_param = np.loadtxt('inputParameters.txt').tolist()
    return i_param[0],i_param[1],i_param[2],i_param[3],i_param[4],i_param[5]

def testClassify(dataTest,W1,W2,B1,B2,igWThreshold,matriksTarget):
    test = []
    test.append(dataTest)
    
    #count TF
    testTF = []
    for word in igWThreshold:
        l = []
        words = []
        for x in test[0].lower().split():
            x = re.sub('[(;:,.\'`?!0123456789)]', '', x)
#            x = stemmer.stem(x.encode('utf-8')) #stemming
            words.append(x.encode('utf-8'))
        c = list(words).count(word)
        l.append(word)
        l.append(c)
        testTF.append(l)
    
    #count IDF
    testIDF = []
    for i in testTF:
        data = []
        counts = 0
        if i[1] > 0:
            counts = counts + 1
        counts = math.log((len(test)+len(testTF))/float(counts+1)) #smoothing
        data.append(i[0])
        data.append(counts)
        testIDF.append(data)
    
    #count TFxIDF
    hasilTFIDF = []
    for i in range(len(testIDF)):
        data = []
        tfidf = 0
        tfidf = testTF[i][1] * testIDF[i][1]
        data.append(testTF[i][0])
        data.append(tfidf)
        hasilTFIDF.append(data)
    
    #feature
    X = []    
    for i in hasilTFIDF:
        X.append(i[1])
        
    #start classify
    label = []
    #feedforward
    maxi = max(X[i])
    mini = min(X[i])
    pp = [(xx-mini)/(maxi-mini) for xx in X[i]]
    p = np.matrix(pp)
#        for j in range(len(p)):
#            p[j] = xmin + (((p[j] - min(p))*(xmax - xmin)) / (max(p) - min(p)))
    V1 = np.dot(p,W1)
    V1 = V1 + B1
    A1 = 1 / (1 + np.exp(-0.01*V1))
    V2 = np.dot(A1,W2)
    V2 = V2 + B2
    A2 = 1 / (1 + np.exp(-0.01 * V2))
    for e in A2:
        for j in range(e.shape[1]):
            if e.item(j) >= 0.5:
                label.append(1)
            else:
                label.append(0)
    
    if label == matriksTarget[0]:
        klasifikasi = 'Informasi'
    elif label == matriksTarget[1]:
        klasifikasi = 'Larangan'
    elif label == matriksTarget[2]:
        klasifikasi = 'Larangan-Informasi'
    elif label == matriksTarget[3]:
        klasifikasi = 'Anjuran'
    elif label == matriksTarget[4]:
        klasifikasi = 'Anjuran-Informasi'
    elif label == matriksTarget[5]:
        klasifikasi = 'Anjuran-Larangan'
    elif label == matriksTarget[6]:
        klasifikasi = 'Anjuran-Larangan-Informasi'
    
    return label,klasifikasi