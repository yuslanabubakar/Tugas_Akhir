# -*- coding: utf-8 -*-
"""
Created on Mon May 21 11:17:15 2018

@author: M. Yuslan Abubakar
"""

import re
import numpy as np
import math
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
#from collections import namedtuple
#from operator import itemgetter

# create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def getData(sheet):
    dataSet = []
    for i in range(1, sheet.max_row+1):
        data = []
        label = []
        data.append(sheet.cell(row=i, column=1).value)
        p = (int(sheet.cell(row=i, column=5).value))
        for i in range(5):
            if i == (p-1):
                label.append(1)
            else:
                label.append(0)
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
            for x in i.lower().split():
                x = re.sub('[(;:,.\'`?!0123456789)]', '', x)
#                x = stemmer.stem(x.encode('utf-8')) #stemming
                if x.encode('utf-8') not in stopwords:
                    textStop.append(x.encode('utf-8'))
        return textStop

    dataStop  = stop_words(data)
    dataStop = np.unique(dataStop)
    thefile = open('preprocessingKitab.txt','w')
    for item in dataStop:
        print>>thefile, item
    print 'Preprocessing file saved'
    return dataStop  

def docEachClass(matriksTarget,dataSet):
    data = []
    thefile = open('docEachClassKitab.txt','w')
    for i in range(len(matriksTarget)):
        d = []
        count = 0
        for x in dataSet:
            if x[1] == matriksTarget[i]:
                count = count + 1
        d.append(matriksTarget[i])
        d.append(count)
        data.append(d)
        print>>thefile, d
    
    return data

def docGivenWord(dataSet,preprocessing,matriksTarget):
    result = []
    for i in preprocessing:
        for j in range(len(matriksTarget)):
            data = []
            countDoc = 0
            for k in dataSet:
                if k[1] == matriksTarget[j]:
                    c = 0
                    words = []
                    for word in k[0].lower().split():
                        word = re.sub('[(;:,.\'`?!0123456789)]', '', word)
#                        word = stemmer.stem(word.encode('utf-8')) #stemming
                        words.append(word.encode('utf-8'))
                    c = list(words).count(i)
                    if c > 0:
                        countDoc = countDoc + 1
            data.append(i)
            data.append(matriksTarget[j])
            data.append(countDoc)
            result.append(data)
    
    thefile = open('FwKitab.txt','w')
    for item in result:
        print>>thefile, item
        
    return result

def probEachClassGivenWord(FW,matriksTarget,docEachClass):
    result = []
    for i in FW:
        for j in range(len(matriksTarget)):
            a = []
            if i[1] == matriksTarget[j]:
                piW = (i[2]+1) / float(docEachClass[j][1]+1) #smoothing
                a.append(i[0])
                a.append(i[1])
                a.append(piW)
                result.append(a)
                
    return result

def informationGain(PIW,dataSet,preprocessing,docEachClass):
    result = []
    thefile = open('informationGainKitab.txt','w')
    for k in preprocessing:
        nilai = 0
        data = []
        for i in PIW:
            if k == i[0]:
                for j in docEachClass:
                    if j[0] == i[1]:
                        nilai += (j[1]/float(len(dataSet))) * -(i[2] * math.log(i[2],2))
        nilai = 1 - nilai
        data.append(k)
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
    tf = []
    for word in IGW:
        hasil = []
        l = []
        doc = 1
        for j in document:
            d = []
            words = []
            for x in j.lower().split():
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
        tf.append(l)
        
    return tf

def idf(TF,dataSet):
    idf = []
    for j in TF:
        d = []
        counts = 0
        for k in j[1]:
            if k[1] > 0:
                counts = counts + 1
        counts = math.log((len(dataSet)+len(TF))/float(counts+1)) #smoothing
        d.append(j[0])
        d.append(counts)
        idf.append(d)
    
    return idf

def tfIDF(TF,dataSet,IDF):
    result = []
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
        result.append(l)
    tfidf = []
    for j in range(len(dataSet)):
        d = []
        li = []
        for k in result:
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

def training(input_p,hidden_p,output_p,lr,epoch,mseStandar,tfidf,dataSet):
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
    B1 = np.matrix(np.random.uniform(0,1,hidden_p))
    np.random.seed(1)
    B2 = np.matrix(np.random.uniform(0,1,output_p))
    np.random.seed(1)
    W1 = np.random.uniform(0,1,(input_p,hidden_p))
    np.random.seed(1)
    W2 = np.random.uniform(0,1,(hidden_p,output_p))
    while (mse > mseStandar and iterate < epoch):
        error = []
        mse = 0
        for i in range(len(X)):
            #feedforward
            p = np.matrix(X[i])
            V1 = np.dot(p,W1)
            V1 = V1 + B1
            A1 = 1 / (1 + np.exp(-0.01*V1))
            V2 = np.dot(A1,W2)
            V2 = V2 + B2
            A2 = 1 / (1 + np.exp(-0.01 * V2)) #sigmoid
#            softmax = lambda x : np.exp(x)/np.sum(np.exp(x))
#            A2 = softmax(V2)
#            result = np.matrix.tolist(A2)
#            maxi = max(result[0])
#            A2 = []
#            for r in result[0]:
#                if r == maxi:
#                    A2.append(1)
#                else:
#                    A2.append(0)
#            A2 = np.matrix(A2)
            e = (dataSet[i][1] - A2)
            error.append(e)
            
            #backpropagation
            D2 = np.multiply(np.multiply(A2,(1-A2)),error[i]) #sigmoid
#            D2 = np.multiply(np.multiply(A2,(np.finfo(float).eps-A2)),error[i]) #softmax
            dw2 = lr * np.dot(np.transpose(D2),A1)
            dw2 = np.transpose(dw2)
            D1 = np.multiply(A1,(1-A1))
            D1 = np.multiply(D1,np.transpose(np.dot(W2,np.transpose(D2))))
            dw1 = lr * np.dot(np.transpose(D1),p)
            dw1 = np.transpose(dw1)
            db1 = lr * D1
            db2 = lr * D2
            W1 = dw1 + W1
            W2 = dw2 + W2
            B1 = db1 + B1
            B2 = db2 + B2
        for e in error:
            mse += sum([e.item(i)**2 for i in range(e.shape[1])])
        mse = mse / len(error)
        iterate += 1
    print 'MSE = ',mse
    print 'Epoch = ',iterate
#    np.savetxt('W1.txt',W1)
#    np.savetxt('W2.txt',W2)
#    np.savetxt('B1.txt',B1)
#    np.savetxt('B2.txt',B2)
    return W1,W2,B1,B2

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
        V1 = np.dot(p,W1)
        V1 = V1 + B1
        A1 = 1 / (1 + np.exp(-0.01*V1))
        V2 = np.dot(A1,W2)
        V2 = V2 + B2
        A2 = 1 / (1 + np.exp(-0.01 * V2)) #sigmoid
        A2 = np.matrix.tolist(A2)
        maxi = max(A2[0])
        for r in A2[0]:
            if r == maxi:
                data.append(1)
            else:
                data.append(0)
#        softmax = lambda x : np.exp(x)/np.sum(np.exp(x))
#        A2 = softmax(V2)
#        A2 = np.matrix.tolist(A2)
#        maxi = max(A2[0])
#        for e in A2[0]:
#            if e == maxi:
#                data.append(1)
#            else:
#                data.append(0)
        label.append(data)
    print 'Finish'
    return label

def akurasi(label,dataSetTest):
    target = []
    for i in dataSetTest:
        target.append(i[1])
        
    benar = 0
    for i in range(len(label)):
        if label[i] == target[i]:
            benar += 1
            
    return (benar / float(len(label))) * 100

def f1_score(matriksTarget,label,dataSetTest):
    truePositif = 0
    falseNegatif = 0
    falsePositif = 0
    trueNegatif = 0
    
    for i in range(len(matriksTarget)):
        for j in range(len(dataSetTest)):
            if dataSetTest[j][1] == matriksTarget[i]:
                if dataSetTest[j][1] == label[j]:
                    truePositif += 1
                else:
                    falseNegatif += 1
            else:
                if label[j] == matriksTarget[i]:
                    falsePositif += 1
                else:
                    trueNegatif += 1
    
    precision = truePositif / float(truePositif + falsePositif)
    recall = truePositif / float(truePositif + falseNegatif)
    return ((2 * precision * recall) / float(precision + recall)) * 100

def getWeights():
    W1 = np.matrix(np.loadtxt('W1.txt'))
    W2 = np.matrix(np.loadtxt('W2.txt'))
    B1 = np.matrix(np.loadtxt('B1.txt'))
    B2 = np.matrix(np.loadtxt('B2.txt'))
    return W1,W2,B1,B2

def getMatriksTarget():
    return np.loadtxt('matriksTargetKitab.txt').tolist()

def getIGWThreshold():
    with open('igThresholdKitab.txt') as f:
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
        klasifikasi = 'Iman'
    elif label == matriksTarget[1]:
        klasifikasi = 'Ilmu'
    elif label == matriksTarget[2]:
        klasifikasi = 'Wudhu'
    elif label == matriksTarget[3]:
        klasifikasi = 'Shalat'
    elif label == matriksTarget[4]:
        klasifikasi = 'Waktu Shalat'
    
    return label,klasifikasi