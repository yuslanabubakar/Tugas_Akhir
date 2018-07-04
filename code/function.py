import re
import numpy as np
import math
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

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
#        first = int(sheet.cell(row=i, column=2).value) * 4
#        second = int(sheet.cell(row=i, column=3).value) * 2
#        third = int(sheet.cell(row=i, column=4).value) * 1
#        p = first + second + third
#        for i in range(7):
#            if i == (p-1):
#                label.append(1)
#            else:
#                label.append(0)
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
        print 'Now Start Remove Stopwords...'
        for i in data:
            for x in i[0].lower().split():
                x = re.sub('[(;:,.\'`?!0123456789)]', '', x)
#                x = stemmer.stem(x.encode('utf-8')) #stemming
                if x.encode('utf-8') not in stopwords:
                    textStop.append(x.encode('utf-8'))
        return textStop

    dataStop  = stop_words(data)
    dataStop = np.unique(dataStop)
    thefile = open('preprocessingMulti.txt','w')
    for item in dataStop:
        print>>thefile, item
    print 'Preprocessing file saved'
    return dataStop    

def docEachClass(matriksTarget,dataSet,dataPre):
    data = []
    thefile = open('docEachClassMulti.txt','w')
    for i in matriksTarget:
        d = []
        count = 0
        for x in dataSet:
            if i == x[1]:
                count = count + 1
        d.append(i)
        d.append(count)
        data.append(d)
        print>>thefile, d
    
    print 'Finish'
    return data

def docGivenWord(matriksTarget,dataPre,dataSet):
    result = []
    for i in dataPre:
        for k in matriksTarget:
            data = []
            countDoc = 0
            for j in dataSet:
                if k == j[1]:
                    c = 0
                    words = []
                    for word in j[0][0].lower().split():
                        word = re.sub('[(;:,.\'`?!0123456789)]', '', word)
#                        word = stemmer.stem(word.encode('utf-8')) #stemming
                        words.append(word.encode('utf-8'))
                    c = list(words).count(i)
                    if c > 0:
                        countDoc = countDoc + 1
            data.append(i)
            data.append(k)
            data.append(countDoc)
            result.append(data)
    thefile = open('FwMulti.txt','w')
    for item in result:
        print>>thefile, item
    print 'Finish'
    return result

def probEachClassGivenWord(Fw,docEachClass,matriksTarget):
    result = []
    for i in Fw:
        for j in docEachClass:
            a = []
            if i[1] == j[0]:
                piW = (i[2]+1) / float(j[1]+1) #smoothing
                a.append(i[0])
                a.append(i[1])
                a.append(piW)
                result.append(a)
    print 'Finish'
    return result

def informationGain(docEachClass,dataSet,piW,dataPre):
    result = []
    thefile = open('informationGainMulti.txt','w')
    for k in dataPre:
        nilai = 0
        data = []
        for i in piW:
            if k == i[0]:
                for j in docEachClass:
                    if j[0] == i[1]:
#                        nilai = nilai + ((j[1] / float(len(dataSet))) * (-(i[2] * math.log(i[2],2))))
                        nilai = nilai + ((j[1] / float(len(dataSet))) * (- i[2]*np.log2(i[2]) - (1 - i[2])*np.log2((1 - i[2]))))
        nilai = 1 - nilai
        data.append(k)
        data.append(nilai)
        result.append(data)
        print>>thefile,data
    print 'Finish'
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
    
def tf(igWThreshold, dataSet):
    document = []
    for i in dataSet:
        document.append(i[0])
    result = []
    for word in igWThreshold:
        hasil = []
        l = []
        doc = 1
        for i in document:
            data = []
            words = []
            for x in i[0].lower().split():
                x = re.sub('[(;:,.\'`?!0123456789)]', '', x)
#                x = stemmer.stem(x.encode('utf-8')) #stemming
                words.append(x.encode('utf-8'))
            c = list(words).count(word)
            data.append(doc)
            data.append(c)
            hasil.append(data)
            doc = doc + 1
        l.append(word)
        l.append(hasil)
        result.append(l)
    print 'Finish'
    return result

def idf(tf,dataSet):
    result = []
    for i in tf:
        data = []
        counts = 0
        for j in i[1]:
            if j[1] > 0:
                counts = counts + 1
        counts = math.log((len(dataSet)+len(tf))/float(counts+1)) #smoothing
        data.append(i[0])
        data.append(counts)
        result.append(data)
    print 'Finish'
    return result

def tfIDF(tf,idf,dataSet):
    result = []
    for i in range(len(tf)):
        tfidf = 0
        d = []
        l = []
        for j in tf[i][1]:
            data = []
            tfidf = j[1] * idf[i][1]
            data.append(j[0])
            data.append(tfidf)
            d.append(data)
        l.append(tf[i][0])
        l.append(d)
        result.append(l)
    tfidf = []
    for i in range(len(dataSet)):
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
        tfidf.append(l)
    result = tfidf
    print 'Finish'
    return result

def training(input_p,hidden_p,output_p,lr,epoch,mseStandar,tfIDF,dataSet):
    X = []
    for i in tfIDF:
        data = []
        for j in i[1]:
            data.append(j[1])
        X.append(data)
    
    #start training
    print 'Start training data...'
    mse = 999999999
    iterate = 1
#    xmin = 0.1
#    xmax = 0.9
    B1 = np.matrix(np.random.rand(hidden_p))
    B2 = np.matrix(np.random.rand(output_p))
    W1 = np.random.rand(input_p,hidden_p)
    W2 = np.random.rand(hidden_p,output_p)
    while (mse > mseStandar and iterate < epoch):
#        print iterate
        error = []
        mse = 0
        for i in range(len(X)):
            #feedforward
            p = np.matrix(X[i])
#            for j in range(len(p)):
#                p[j] = xmin + (((p[j] - min(p))*(xmax - xmin)) / (max(p) - min(p)))
            V1 = np.dot(p,W1)
            V1 = V1 + B1
            A1 = 1 / (1 + np.exp(-0.01*V1))
            V2 = np.dot(A1,W2)
            V2 = V2 + B2
            A2 = 1 / (1 + np.exp(-0.01 * V2))
            T = np.matrix(dataSet[i][1])
            e = (T - A2)
            error.append(e)
            
            #backpropagation
            D2 = np.multiply(np.multiply(A2,(1-A2)),error[i])
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
#        print mse
        iterate += 1
    print 'MSE = ',mse
    print 'Epoch = ',iterate
    print 'Finish'
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
    
#    labelClass = len([list(x) for x in set(tuple(x) for x in target)])
    labelClass = 3
    hLoss = (1/float(labelClass)) * (1/float(len(dataSet))) * errorH
    print 'Finish'
    return hLoss

def getWeights():
    W1 = np.matrix(np.loadtxt('W1Multi.txt'))
    W2 = np.matrix(np.loadtxt('W2Multi.txt'))
    B1 = np.matrix(np.loadtxt('B1Multi.txt'))
    B2 = np.matrix(np.loadtxt('B2Multi.txt'))
    return W1,W2,B1,B2

def getMatriksTarget():
    return np.loadtxt('matriksTargetMulti.txt').tolist()

def getIGWThreshold():
    with open('igWThresholdMulti.txt') as f:
        alist = f.read().splitlines(True)
    return alist

def getInputParameters():
    i_param = np.loadtxt('inputParametersMulti.txt').tolist()
    return i_param[0],i_param[1],i_param[2],i_param[3],i_param[4],i_param[5]

def testClassify(dataTest,W1,W2,B1,B2,feature,matriksTarget):
    test = []
    test.append(dataTest)
    
    #count TF
    testTF = []
    for word in feature:
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