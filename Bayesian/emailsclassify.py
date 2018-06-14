from re import sub
from os import listdir
from collections import Counter
from itertools import chain
from numpy import array
from jieba import cut
from sklearn.naive_bayes import MultinomialNB

#存放所有文件中的单词
#每个元素是一个子列表，其中存放一个文件中的单词
allWords = []

def getWordFromFile(txtFile):
    words = []
    with open(txtFile,encoding='utf8') as fp:
        for line in fp:
            line = line.strip()
            #过滤干扰字符或无效字符
            line = sub(r'[.【】0-9、-。！～\*]','',line)
            line = cut(line)
            #过滤长度为1的词
            line = filter(lambda word:len(word) >1,line)
            words.extend(line)
    return words

def getTopNWords(topN):
    #按文件编号顺序处理当前文件夹中所有记事本文件
    #共151封邮件内容，0.txt到126.txt是垃圾邮件内容
    #127.txt到150.txt为正常邮件内容
    txtFiles = [str(i) + '.txt' for i in range(151)]
    for txtFile in txtFiles:
        allWords.append(getWordFromFile(txtFile))
        #获取并返回出现次数最多的前topN个单词
    freq = Counter(chain(*allWords))
    return [w[0] for w in freq.most_common((topN))]

#全部训练集中出现次数最多的前400个单词
topWords = getTopNWords(400)

#获取特征向量，前400个单词的每个单词在每个邮件中出现的频率
vector = []
for words in allWords:
    temp = list(map(lambda x: words.count(x),topWords))
    vector.append(temp)
vector = array(vector)
labels = array([1]*127 + [0]*24)

#创建模型，使用已知训练集进行训练
model = MultinomialNB
model.fit(vector,labels)

#使用训练好的模型对未知邮件内容进行分类
def predict(txtFile):
    #获取制定邮件文件内容，返回分类结果
    words = getWordFromFile(txtFile)
    currentVector = array(tuple(map(lambda x: words.count(x),topWords)))
    result = model.predict(currentVector.reshape(1,-1))
    return '垃圾邮件' if result == 1 else '正常邮件'

print(predict('151.txt'))
print(predict('152.txt'))
print(predict('153.txt'))
print(predict('154.txt'))
print(predict('155.txt'))