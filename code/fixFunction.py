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

# create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def getData(sheet):
    Data = namedtuple('Data','anjuran, larangan, informasi')
    dataInfo = []
    dataAnjur = []
    dataLarang = []
    for i in range(1, sheet.max_row+1):
        if (int(sheet.cell(row=i, column=2).value)) == 1:
            dataAnjur.append(sheet.cell(row=i, column=1).value)
        if (int(sheet.cell(row=i, column=3).value)) == 1:
            dataLarang.append(sheet.cell(row=i, column=1).value)
        if (int(sheet.cell(row=i, column=4).value)) == 1:
            dataInfo.append(sheet.cell(row=i, column=1).value)

    return Data(dataAnjur,dataLarang,dataInfo)

def preprocessing(data):
    Data = namedtuple('Data','anjuran,larangan,informasi')
    f = open('stopwords.txt','r')
    stopwords = f.read()

    def stop_words(data):
        textStop = []
        for i in data:
            for x in i.lower().split():
                x = re.sub('[(;:,.\'`?!0123456789)]', '', x)
#                x = stemmer.stem(x.encode('utf-8')) #stemming
                if x.encode('utf-8') not in stopwords:
                    textStop.append(x.encode('utf-8'))
        return textStop

    anjur = np.unique(stop_words(data.anjuran))
    larang = np.unique(stop_words(data.larangan))
    info = np.unique(stop_words(data.informasi))
#    thefile = open('dataStop.txt','w')
#    for item in dataStop:
#        print>>thefile, item
#    print 'Preprocessing file saved'
    return Data(anjur,larang,info)  

def docEachClass(dataAnjur,dataLarang,dataInfo):
    data = []
    d = []
    d.append('anjuran')
    d.append(len(dataAnjur))
    data.append(d)
    d = []
    d.append('larangan')
    d.append(len(dataLarang))
    data.append(d)
    d = []
    d.append('informasi')
    d.append(len(dataInfo))
    data.append(d)
    
    print 'Finish'
    return data

def docGivenWord(data,preprocessing):
    Data = namedtuple('Data','anjuran, larangan, informasi')
    FW = []
    for i in range(len(data)):
        fw = []
        for j in preprocessing[i]:
            d = []
            countDoc = 0
            for k in data[i]:
                c = 0
                words = []
                for word in k.lower().split():
                    word = re.sub('[(;:,.\'`?!0123456789)]', '', word)
#                    word = stemmer.stem(word.encode('utf-8')) #stemming
                    words.append(word.encode('utf-8'))
                c = list(words).count(j)
                if c > 0:
                    countDoc = countDoc + 1
            d.append(j)
            d.append(countDoc)
            fw.append(d)
        FW.append(fw)
        
    return Data(FW[0],FW[1],FW[2])

def probEachClassGivenWord(FW, data):
    Data = namedtuple('Data','anjuran, larangan, informasi')
    PIW = []
    for i in range(len(data)):
        piw = []
        for j in FW[i]:
            a = []
            p = (j[1]) / float(len(data[i]))
            a.append(j[0])
            a.append(p)
            piw.append(a)
        PIW.append(piw)
        
    return Data(PIW[0],PIW[1],PIW[2])

def informationGain(PIW):
    Data = namedtuple('Data','anjuran, larangan, informasi')
    IGW = []
    for i in range(len(PIW)):
        ig = []
        for j in PIW[i]:
            nilai = 0
            data = []
            nilai = (-(j[1] * math.log(j[1],2)))
            data.append(j[0])
            data.append(nilai)
            ig.append(data)
        IGW.append(ig)
    
    return Data(IGW[0],IGW[1],IGW[2])

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
    
def tf(IGW,data):
    Data = namedtuple('Data','anjuran, larangan, informasi')
    TF = []
    for i in range(len(IGW)):
        tf = []
        for word in IGW[i]:
            hasil = []
            l = []
            doc = 1
            for j in data[i]:
                d = []
                words = []
                for x in j.lowe().split():
                    x = re.sub('[(;:,.\'`?!0123456789)]', '', x)
#                    x = stemmer.stem(x.encode('utf-8')) #stemming
                    words.append(x.encode('utf-8'))
                c = list(words).count(word)
                d.append(doc)
                d.append(c)
                hasil.append(d)
                doc = doc + 1
            l.append(word)
            l.append(hasil)
            tf.append(l)
        TF.append(tf)
        
    return Data(TF[0],TF[1],TF[2])

def idf(tfAnjur,tfLarang,tfInfo,dataAnjur,dataLarang,dataInfo):
    idfAnjur = []
    idfLarang = []
    idfInfo = []
    for i in tfAnjur:
        data = []
        counts = 0
        for j in i[1]:
            if j[1] > 0:
                counts = counts + 1
        counts = math.log((len(dataAnjur)+len(tfAnjur))/float(counts+1)) #smoothing
        data.append(i[0])
        data.append(counts)
        idfAnjur.append(data)
        
    for i in tfLarang:
        data = []
        counts = 0
        for j in i[1]:
            if j[1] > 0:
                counts = counts + 1
        counts = math.log((len(dataLarang)+len(tfLarang))/float(counts+1)) #smoothing
        data.append(i[0])
        data.append(counts)
        idfLarang.append(data)
        
    for i in tfInfo:
        data = []
        counts = 0
        for j in i[1]:
            if j[1] > 0:
                counts = counts + 1
        counts = math.log((len(dataInfo)+len(tfInfo))/float(counts+1)) #smoothing
        data.append(i[0])
        data.append(counts)
        idfInfo.append(data)
    print 'Finish'
    return idfAnjur,idfLarang,idfInfo

def tfIDF(tfAnjur,tfLarang,tfInfo,dataAnjur,dataLarang,dataInfo,idfAnjur,idfLarang,idfInfo):
    result = []
    for i in range(len(tfAnjur)):
        tidfAnjur = 0
        d = []
        l = []
        for j in tfAnjur[i][1]:
            data = []
            tidfAnjur = j[1] * idfAnjur[i][1]
            data.append(j[0])
            data.append(tidfAnjur)
            d.append(data)
        l.append(tfAnjur[i][0])
        l.append(d)
        result.append(l)
    tfidfAnjur = []
    for i in range(len(dataAnjur)):
        d = []
        l = []
        for j in result:
            data = []
            for k in j[1]:
                if i+1 == k[0]:
                    data.append(j[0])
                    data.append(k[1])
            d.append(data)
        l.append(i+1)
        l.append(d)
        tfidfAnjur.append(l)
        
    result = []
    for i in range(len(tfLarang)):
        tidfLarang = 0
        d = []
        l = []
        for j in tfLarang[i][1]:
            data = []
            tidfLarang= j[1] * idfLarang[i][1]
            data.append(j[0])
            data.append(tidfLarang)
            d.append(data)
        l.append(tfLarang[i][0])
        l.append(d)
        result.append(l)
    tfidfLarang= []
    for i in range(len(dataLarang)):
        d = []
        l = []
        for j in result:
            data = []
            for k in j[1]:
                if i+1 == k[0]:
                    data.append(j[0])
                    data.append(k[1])
            d.append(data)
        l.append(i+1)
        l.append(d)
        tfidfLarang.append(l)
        
    result = []
    for i in range(len(tfInfo)):
        tidfInfo = 0
        d = []
        l = []
        for j in tfInfo[i][1]:
            data = []
            tidfInfo = j[1] * idfInfo[i][1]
            data.append(j[0])
            data.append(tidfInfo)
            d.append(data)
        l.append(tfInfo[i][0])
        l.append(d)
        result.append(l)
    tfidfInfo = []
    for i in range(len(dataInfo)):
        d = []
        l = []
        for j in result:
            data = []
            for k in j[1]:
                if i+1 == k[0]:
                    data.append(j[0])
                    data.append(k[1])
            d.append(data)
        l.append(i+1)
        l.append(d)
        tfidfInfo.append(l)
    print 'Finish'
    return tfidfAnjur,tfidfLarang,tfidfInfo

def trainingAnjur(input_p,hidden_p,output_p,lr,epoch,mseStandar,tfidfAnjur):
    X = []
    for i in tfidfAnjur:
        data = []
        for j in i[1]:
            data.append(j[1])
        X.append(data)
    
    #start training
    mse = 999999999
    iterate = 1
    B1Anjur = np.matrix(np.random.rand(hidden_p))
    B2Anjur = np.matrix(np.random.rand(output_p))
    W1Anjur = np.random.rand(input_p,hidden_p)
    W2Anjur = np.random.rand(hidden_p,output_p)
    while (mse > mseStandar and iterate <= epoch):
        error = []
        mse = 0
        for i in range(len(X)):
            #feedforward
            p = np.matrix(X[i])
            V1 = np.dot(p,W1Anjur)
            V1 = V1 + B1Anjur
            A1 = 1 / (1 + np.exp(-0.1*V1))
            V2 = np.dot(A1,W2Anjur)
            V2 = V2 + B2Anjur
            A2 = 1 / (1 + np.exp(-0.1 * V2))
            e = (1 - A2) #targetnya selalu 1
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
    
def trainingLarang(input_p,hidden_p,output_p,lr,epoch,mseStandar,tfidfLarang):
    X = []
    for i in tfidfLarang:
        data = []
        for j in i[1]:
            data.append(j[1])
        X.append(data)
    
    #start training
    mse = 999999999
    iterate = 1
    B1Larang= np.matrix(np.random.rand(hidden_p))
    B2Larang = np.matrix(np.random.rand(output_p))
    W1Larang = np.random.rand(input_p,hidden_p)
    W2Larang = np.random.rand(hidden_p,output_p)
    while (mse > mseStandar and iterate <= epoch):
        error = []
        mse = 0
        for i in range(len(X)):
            #feedforward
            p = np.matrix(X[i])
            V1 = np.dot(p,W1Larang)
            V1 = V1 + B1Larang
            A1 = 1 / (1 + np.exp(-0.1*V1))
            V2 = np.dot(A1,W2Larang)
            V2 = V2 + B2Larang
            A2 = 1 / (1 + np.exp(-0.1 * V2))
            e = (1 - A2) #targetnya selalu 1
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

def trainingInfo(input_p,hidden_p,output_p,lr,epoch,mseStandar,tfidfInfo):
    X = []
    for i in tfidfInfo:
        data = []
        for j in i[1]:
            data.append(j[1])
        X.append(data)
    
    #start training
    mse = 999999999
    iterate = 1
    B1Info= np.matrix(np.random.rand(hidden_p))
    B2Info = np.matrix(np.random.rand(output_p))
    W1Info = np.random.rand(input_p,hidden_p)
    W2Info = np.random.rand(hidden_p,output_p)
    while (mse > mseStandar and iterate <= epoch):
        error = []
        mse = 0
        for i in range(len(X)):
            #feedforward
            p = np.matrix(X[i])
            V1 = np.dot(p,W1Info)
            V1 = V1 + B1Info
            A1 = 1 / (1 + np.exp(-0.1*V1))
            V2 = np.dot(A1,W2Info)
            V2 = V2 + B2Info
            A2 = 1 / (1 + np.exp(-0.1 * V2))
            e = (1 - A2) #targetnya selalu 1
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
        data = []
        p = np.matrix(X[i])
#        for j in range(len(p)):
#            p[j] = xmin + (((p[j] - min(p))*(xmax - xmin)) / (max(p) - min(p)))
        V1 = np.dot(p,W1)
        V1 = V1 + B1
        A1 = 1 / (1 + np.exp(-1*V1))
        V2 = np.dot(A1,W2)
        V2 = V2 + B2
        A2 = 1 / (1 + np.exp(-1 * V2))
        for e in A2:
            for j in range(e.shape[1]):
                if e.item(j) >= 0.5:
                    data.append(1)
                else:
                    data.append(0)
        label.append(data)
    print 'Finish'
    return label
        
def hammingLoss(label,dataSet):
    print 'Count hamming loss...'
    target = []
    for i in dataSet:
        target.append(i[1])
    
    errorH = 0
    for i in range(len(target)):
        e = np.array(label[i]) - np.array(target[i])
        for err in e:
            errorH += err**2
    
    labelClass = len([list(x) for x in set(tuple(x) for x in target)])
    hLoss = (1/float(labelClass)) * (1/float(len(dataSet))) * errorH
    print 'Finish'
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
#                x = stemmer.stem(x.encode('utf-8')) #stemming
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
    p = np.matrix(X)
#        for j in range(len(p)):
#            p[j] = xmin + (((p[j] - min(p))*(xmax - xmin)) / (max(p) - min(p)))
    V1 = np.dot(p,W1)
    V1 = V1 + B1
    A1 = 1 / (1 + np.exp(-1*V1))
    V2 = np.dot(A1,W2)
    V2 = V2 + B2
    A2 = 1 / (1 + np.exp(-1 * V2))
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