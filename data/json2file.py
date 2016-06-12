#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json

collectAll = []
trainContext = []
trainOpinion = []
testContext = []
testOpinion = []
BLANK = "__BLANK__"
with open('train.json', 'r') as trainFile:
    trainLines = json.load(trainFile)
    print ("[*] Reading train.json")
    for oneLine in trainLines:
        trainContext.append(oneLine["text"])
        collectAll.append(oneLine["text"])
        lineQ = oneLine["question"]
        trainOpinion.append(str(oneLine["correct"]))
        for op in oneLine["options"]:
            cpQ = lineQ.replace(BLANK, op)
            trainOpinion.append(cpQ)
            collectAll.append(cpQ)
print ("[*] Reading done")

with open('train.context', 'w') as f:
    print ("[*] Writing train.context")
    for line in trainContext:
        f.write(line.encode('utf-8'))
        f.write('\n')
print ("[*] Writing train.context done")

with open('train.opinion', 'w') as f:
    print ("[*] Writing train.opinion")
    for line in trainOpinion:
        f.write(line.encode('utf-8'))
        f.write('\n')
print ("[*] Writing train.opinion done")


with open('test.json', 'r') as testFile:
    testLines = json.load(testFile)
    print ("[*] Reading test.json")
    for oneLine in testLines:
        testContext.append(oneLine["text"])
        collectAll.append(oneLine["text"])
        lineQ = oneLine["question"]
        testOpinion.append(str(oneLine["correct"]))
        for op in oneLine["options"]:
            cpQ = lineQ.replace(BLANK, op)
            testOpinion.append(cpQ)
            collectAll.append(cpQ)
print ("[*] Reading done")

with open('test.context', 'w') as f:
    print ("[*] Writing test.context")
    for line in testContext:
        f.write(line.encode('utf-8'))
        f.write('\n')
print ("[*] Writing test.context done")

with open('test.opinion', 'w') as f:
    print ("[*] Writing test.opinion")
    for line in testOpinion:
        f.write(line.encode('utf-8'))
        f.write('\n')
print ("[*] Writing test.opinion done")

with open('collect.data', 'w') as f:
    print ("[*] Writing collect.data")
    for line in collectAll:
        f.write(line.encode('utf-8'))
        f.write('\n')
