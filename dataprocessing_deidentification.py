

import numpy as np
import os
import pandas as pd
import nltk
import pycrfsuite
os.chdir('C:/Users/Anurag/Desktop/data mining/anurag/project')


f = open('deid_surrogate_train_all_version2.xml')
xml = f.read() 
from bs4 import BeautifulSoup
soup = BeautifulSoup(xml, 'xml')


from nltk import word_tokenize
from nltk import sent_tokenize



sentence = str(soup).rstrip().split('\n')

sentence = sentence[15:52614]

del sentence[25419]


wordlist = []
for sent in sentence:
    if ("<PHI" in sent) == True:
       while "<PHI" in sent: 
           t = sent[sent.find("<PHI") : sent.find("PHI>")+4] 
           f =t[t.find('=')+1 : t.find('>')],t[t.find('>')+1:len(t)-6]
           d = (f[0][1:-1] +"$" +f[1])
           d = d.replace(" " ,"-")
           d = d.replace(",","-")
           sent = sent.replace(t,d)
       h = sent.split()
       wordlist.append(h)
    else:
        wordlist.append(sent.split())
        

lo = []
for sent in wordlist:
   k = [tuple((j.replace("$",",")).split(",") ) if "$" in j else ("other",j) for j in sent ]
   lo.append(k)
   
#del lo[25434]  
jhgf = []
for sent in lo:
    k = [i[1] for i in sent]
    jhgf.append(k)
    



#==============================================================================
# asas =[]
# count = 0
# for lis in jhgf:
#     if "" in lis:
#        listbad = lis
#        asas.append(listbad)
# 
# for lis in jhgf:
#     if "" in lis:
#         lis.remove('')
#==============================================================================
for lis in jhgf:
     if "" in lis:
        lis.remove('')
        
asas =[]
count = 0
for lis in jhgf:
     if "" in lis:
        listbad = lis
        asas.append(listbad)
        count = count + 1

      

        
from nltk.tag.perceptron import PerceptronTagger
tagger = PerceptronTagger()

tagset = None

tags = [nltk.tag._pos_tag(i, tagset, tagger) for i in jhgf]



b =[]
for sent in lo:
  a=[]
  for i in sent:
        s = list(i)
        s[0],s[1] = s[1],s[0]
        i = tuple(s)
        a.append(i)
  b.append(a)
  
#del lo[25434] 
  
#==============================================================================
# asas =[]
# count = 0
# for lis in b:
#     for i in lis:
#         if i[0] == "":
#            lis.remove(i)
# 
# asas =[]
# count = 0
# for lis in b:
#     for i in lis:
#         if i[0] == "":
#            asas.append(i)
# 
# count = 0
# for i,j in zip(b,tags):
#     if len(i) != len(j):
#         count = count + 1
#==============================================================================
#   s =[]
# count =0
# for i,j in zip(b,tags):
#     d =[i[k][0] != j[k][0] for k in range(len(i))]
#     if True in d:
#         count = count+1
#     s.append(d)
#   
#==============================================================================
# # 
  
#==============================================================================
from collections import OrderedDict

count = 0
final =[]
for i,j in zip(b,tags):
    lst1 = i
    lst2 = j
    dict1 = OrderedDict(lst1)
    dict2 = OrderedDict(lst2)
    lst3 = [(k, dict1[k], dict2[k]) for k in dict1]
    final.append(lst3)
    count = count+1
    print count
count = 0      
for i in lo :
    for j in i:
        if len(j)>2:
           print j
           print lo.index(i)
           count = count+1
           
###del line 434
train = []
for sent in final:
    d =[]
    for word in sent:
        s = list(word)
        if (s[1] == 'HOSPITAL') | (s[1] == 'DOCTOR') | (s[1] == 'PATIENT'):
            s[0] = s[0].replace("-" , " ")
        else:
            s[0] = s[0]
        word = tuple(s)
        d.append(word)
    train.append(d)
    

# adding paragraph level features
import re
pattern = r'[(A-Z)+\s]+:'
    
s =[]
for i in range(len(sentence)):
    d = re.findall(pattern,sentence[i])
    if len(d)!= 0:
        d[0] = d[0].replace("(","")
        d[0] = d[0].replace(")" , "")
        d[0] = d[0].lstrip()
        tup = (d,i)
        if tup[0][0] != ":" :
           s.append(tup)
    
    
# extrapolating the missing paragraph desc        
sss =[]
j =0
for i in range(len(s)):
     if i < len(s):
         for k in range(s[i][1] -s[j][1]) :
             sss.append(s[j])
         j=i
sss.append(s[len(s)-1])    
    
#creating a single list out of two lis 
abcsd =[] 
for i in sss:
    a = i[0]
    b= i[1]
    c =[a,b]
    abcsd.append(c)  
    
ran = np.arange(abcsd[0][1],abcsd[len(abcsd)-1][1] + 1)

dass =[]    
for i in abcsd:
   a= i[0]
   b =i[1]
   c = [b]
   sd = [a,c]
   asp = sum(sd,[])
   dass.append(asp)
   
    
rew = []   
for a,b in zip(dass,ran):  
      x =[a,b]
      rew.append(x)
passs = []
for j in rew:
      a = [j[0][0]]
      b = [j[0][1]]
      c = [j[1]]
      asd = [a,b,c]
      esd = sum(asd,[])
      passs.append(esd)
      
 #join the data and para dist  
added = []
for i,j in zip(train,passs):
     a =  i
     b = j
     c = [i,j]
     added.append(c)
final_train = []     
for i in added:
    f =[]
    for j in i[0]:
       a,b,c = j
       d = i[1][0]
       h = d.replace(" :","")
       e = (a,b,c,h)
       f.append(e)
    final_train.append(f)
    
    
####################test data prep#################################
    
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 20:42:34 2015

@author: prashanth
"""

import numpy as np
import os
import pandas as pd
import nltk
import pycrfsuite
os.chdir('C:/Users/Anurag/Desktop/data mining/anurag/project')

f = open('deid_surrogate_test_all_groundtruth_version2.xml')
xml = f.read() 
from bs4 import BeautifulSoup
soup = BeautifulSoup(xml, 'xml')


from nltk import word_tokenize
from nltk import sent_tokenize



sentence = str(soup).rstrip().split('\n')

sentence = sentence[18:18969]

del sentence[25419]


wordlist = []
for sent in sentence:
    if ("<PHI" in sent) == True:
       while "<PHI" in sent: 
           t = sent[sent.find("<PHI") : sent.find("PHI>")+4] 
           f =t[t.find('=')+1 : t.find('>')],t[t.find('>')+1:len(t)-6]
           d = (f[0][1:-1] +"$" +f[1])
           d = d.replace(" " ,"-")
           d = d.replace(",","-")
           sent = sent.replace(t,d)
       h = sent.split()
       wordlist.append(h)
    else:
        wordlist.append(sent.split())
        

lo = []
for sent in wordlist:
   k = [tuple((j.replace("$",",")).split(",") ) if "$" in j else ("other",j) for j in sent ]
   lo.append(k)
   
#del lo[25434]  
jhgf = []
for sent in lo:
    k = [i[1] for i in sent]
    jhgf.append(k)
    



#==============================================================================
# asas =[]
# count = 0
# for lis in jhgf:
#     if "" in lis:
#        listbad = lis
#        asas.append(listbad)
# 
# for lis in jhgf:
#     if "" in lis:
#         lis.remove('')
#==============================================================================
for lis in jhgf:
     if "" in lis:
        lis.remove('')
        
asas =[]
count = 0
for lis in jhgf:
     if "" in lis:
        listbad = lis
        asas.append(listbad)
        count = count + 1

      

        
from nltk.tag.perceptron import PerceptronTagger
tagger = PerceptronTagger()

tagset = None

tags = [nltk.tag._pos_tag(i, tagset, tagger) for i in jhgf]

b =[]
for sent in lo:
  a=[]
  for i in sent:
        s = list(i)
        s[0],s[1] = s[1],s[0]
        i = tuple(s)
        a.append(i)
  b.append(a)
  
#del lo[25434] 
  
#==============================================================================
# asas =[]
# count = 0
# for lis in b:
#     for i in lis:
#         if i[0] == "":
#            lis.remove(i)
# 
# asas =[]
# count = 0
# for lis in b:
#     for i in lis:
#         if i[0] == "":
#            asas.append(i)
# 
# count = 0
# for i,j in zip(b,tags):
#     if len(i) != len(j):
#         count = count + 1
#==============================================================================
#   s =[]
# count =0
# for i,j in zip(b,tags):
#     d =[i[k][0] != j[k][0] for k in range(len(i))]
#     if True in d:
#         count = count+1
#     s.append(d)
#   
#==============================================================================
# # 
  
#==============================================================================
from collections import OrderedDict

count = 0
final =[]
for i,j in zip(b,tags):
    lst1 = i
    lst2 = j
    dict1 = OrderedDict(lst1)
    dict2 = OrderedDict(lst2)
    lst3 = [(k, dict1[k], dict2[k]) for k in dict1]
    final.append(lst3)
    count = count+1
    print count
count = 0      
for i in lo :
    for j in i:
        if len(j)>2:
           print j
           print lo.index(i)
           count = count+1
           
###del line 434
test = []
for sent in final:
    d =[]
    for word in sent:
        s = list(word)
        if (s[1] == 'HOSPITAL') | (s[1] == 'DOCTOR') | (s[1] == 'PATIENT'):
            s[0] = s[0].replace("-" , " ")
        else:
            s[0] = s[0]
        word = tuple(s)
        d.append(word)
    test.append(d)
    

# adding paragraph level features
import re
pattern = r'[(A-Z)+\s]+:'
    
s =[]
for i in range(len(sentence)):
    d = re.findall(pattern,sentence[i])
    if len(d)!= 0:
        d[0] = d[0].replace("(","")
        d[0] = d[0].replace(")" , "")
        d[0] = d[0].lstrip()
        tup = (d,i)
        if tup[0][0] != ":" :
           s.append(tup)
    
    
# extrapolating the missing paragraph desc        
sss =[]
j =0
for i in range(len(s)):
     if i < len(s):
         for k in range(s[i][1] -s[j][1]) :
             sss.append(s[j])
         j=i
sss.append(s[len(s)-1])    
    
#creating a single list out of two lis 
abcsd =[] 
for i in sss:
    a = i[0]
    b= i[1]
    c =[a,b]
    abcsd.append(c)  
    
ran = np.arange(abcsd[0][1],abcsd[len(abcsd)-1][1] + 1)

dass =[]    
for i in abcsd:
   a= i[0]
   b =i[1]
   c = [b]
   sd = [a,c]
   asp = sum(sd,[])
   dass.append(asp)
   
    
rew = []   
for a,b in zip(dass,ran):  
      x =[a,b]
      rew.append(x)
passs = []
for j in rew:
      a = [j[0][0]]
      b = [j[0][1]]
      c = [j[1]]
      asd = [a,b,c]
      esd = sum(asd,[])
      passs.append(esd)
      
 #join the data and para dist  
added = []
for i,j in zip(test,passs):
     a =  i
     b = j
     c = [i,j]
     added.append(c)
final_test = []     
for i in added:
    f =[]
    for j in i[0]:
       a,b,c = j
       d = i[1][0]
       h = d.replace(" :","")
       e = (a,b,c,h)
       f.append(e)
    final_test.append(f)
               
           
    
    
    
###############data processing is done ############################  
    
def ifdate(word):
    a = 0
    try:
        DateRegex1 = re.compile(r'\d\d/\d\d/*[0-9]*')
        mo1 = DateRegex1.search(word)
        a = len(mo1.group())
    
    except:
        pass
    
    try:
        DateRegex2 = re.compile(r'([0-9]{1,2}th)')
        mo2 = DateRegex2.search(word)
        a =len(mo2.group())
    
    except:
        pass
    
    try:
    
        DateRegex3 = re.compile(r'([0-9]{1,2}st)')
        mo3 = DateRegex3.search(word)
        a =len(mo3.group())
    
    except:
        pass
        
    x = (a > 0)
    
    return  x
    
    
def ifphone(word):
    
    a = 0
    try:
        phoneNumRegex1 = re.compile(r'\d\d\d-\d\d\d-\d\d\d\d')
        mo1 = phoneNumRegex1.search(word)
        a = len(mo1.group())

    
    except:
        pass
    
    try:
        phoneNumRegex2 = re.compile(r'\(\d\d\d\)-\d\d\d-\d\d\d\d')
        mo2 = phoneNumRegex2.search(word)
        a = len(mo2.group())
    
    except:
        pass
    
    try:
    
       phoneNumRegex3 = re.compile(r'\d\d\d-\d\d\d\d')
       mo3 = phoneNumRegex3.search(word)
       a = len(mo3.group())
    
    except:
        pass
        
    x = (a > 0)
    
    return x
    

def ifspace(word):
    
   x = word.find(" ") > 0
   
   return x
   
def Wordcaps(word):
    
    words=word.split()
    if len(words) == 1:
    
        x  = (words[0].isupper()) 
    else:
    
        x  = (words[0].isupper()) & (words[1].isupper())
       
    return x
    
    
def Wordcaps2(word):
    
    words=word.split()
    
    if len(words) == 1:
    
        x  = (words[0].isupper()) 
    else:
        x  = (words[0].isupper()) | (words[1].isupper())
       
    return x
    

    
###############Feature creation######################################    
    
    
    
    
def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][2]
    paragraph = sent[i][3]
    features = [
        'bias',
        'word.lower=' + word.lower(),
        'word[-3:]=' + word[-3:],
        'word[-2:]=' + word[-2:],
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % word.isdigit(),
        'Wordcaps2=%s' % Wordcaps2(word),
        'Wordcaps=%s' % Wordcaps(word),
        'ifspace=%s' % ifspace(word),
        'ifphone=%s' % ifphone(word),
        'ifdate=%s' % ifdate(word),



        'postag=' + postag,
        'postag[:2]=' + postag[:2],
        'paragraph=' + paragraph,
        'paragraph[:2]=' + paragraph[:2],
    ]
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][2]
        paragraph1 = sent[i-1][3]
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isupper=%s' % word1.isupper(),
            '-1:Wordcaps2=%s' % Wordcaps2(word1),
            '-1:Wordcaps=%s' % Wordcaps(word1),
            '-1:ifspace=%s' % ifspace(word1),
            '-1:ifphone=%s' % ifphone(word1),
            '-1:ifdate=%s' % ifdate(word1),

            '-1:postag=' + postag1,
            '-1:postag[:2]=' + postag1[:2],
            '-1:paragraph=' + paragraph1,
            '-1:paragraph[:2]=' + paragraph1[:2],
        ])
    else:
        features.append('BOS')
        
        
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        paragraph1 = sent[i+1][3]
        features.extend([
            '+1:word.lower=' + word1.lower(),
            '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.isupper=%s' % word1.isupper(),
            '+1:postag=' + postag1,
            '+1:postag[:2]=' + postag1[:2],
            '+1:paragraph=' + paragraph1,
            '+1:paragraph[:2]=' + paragraph1[:2],
        ])
    else:
        features.append('EOS')
                
    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token,label,postag,paragraph in sent]

def sent2tokens(sent):
    return [token for token,label,postag,paragraph in sent] 
    
    
%%time
X_train = [sent2features(s) for s in final_train]
y_train = [sent2labels(s) for s in final_train]

X_test = [sent2features(s) for s in final_test]
y_test = [sent2labels(s) for s in final_test]


%%time
trainer = pycrfsuite.Trainer(verbose=False)

for xseq, yseq in zip(X_train, y_train):
    trainer.append(xseq, yseq)
    
c1seq = np.arange(0,1.2,0.1)    
c2seq = np.arange(1e-6,1,1e-1)
accu_list = []
for i,j in zip(c1seq,c2seq):
    trainer.set_params({
    'c1': 0.5,   # coefficient for L1 penalty
    'c2': 1e-3,  # coefficient for L2 penalty
    'max_iterations': 50,  # stop earlier

    # include transitions that are possible, but not observed
    'feature.possible_transitions': True,
    })

    trainer.params()
    trainer.train('ner.crfsuite')
    !ls -lh ./ner.crfsuite
    tagger = pycrfsuite.Tagger()
    tagger.open('ner.crfsuite')
    acc = (i,j,accuracy(final_test))
    accu_list.append(acc)


def accuracy(final_test):
    correct_pred = 0
    incorrect_pred = 0
    correct_hosp  = 0
    correct_pat = 0
    correct_doc = 0
    incorrect_hosp  = 0
    incorrect_pat = 0
    incorrect_doc = 0
    correct_date = 0
    incorrect_date = 0
    correct_phone = 0
    incorrect_phone = 0
    correct_ID = 0
    incorrect_ID = 0
    for sent in final_test:
         pred  =tagger.tag(sent2features(sent))
         actual = sent2labels(sent)
         for k in range(len(pred)):
             if pred[k] == actual[k] :
                 correct_pred = correct_pred + 1
             else:
                 incorrect_pred = incorrect_pred + 1
             
             if (actual[k] == "HOSPITAL"):
                 if (pred[k] == actual[k]):
                      correct_hosp = correct_hosp + 1
                 else:
                     incorrect_hosp = incorrect_hosp + 1
             if (actual[k] == "PATIENT"):
                 if (pred[k] == actual[k]):
                      correct_pat = correct_pat + 1
                 else:
                     incorrect_pat = incorrect_pat + 1
             if (actual[k] == "DOCTOR"):
                 if (pred[k] == actual[k]):
                      correct_doc = correct_doc + 1
                 else:
                     incorrect_pat = incorrect_pat + 1  
             if (actual[k] == "DATE"):
                 if (pred[k] == actual[k]):
                      correct_date = correct_date + 1
                 else:
                     incorrect_date = incorrect_date + 1
                     
             if (actual[k] == "PHONE"):
                 if (pred[k] == actual[k]):
                      correct_phone = correct_phone + 1
                 else:
                     incorrect_phone = incorrect_phone + 1  
            
             if (actual[k] == "ID"):
                 if (pred[k] == actual[k]):
                      correct_ID = correct_ID + 1
                 else:
                     incorrect_ID = incorrect_ID + 1

 
    total = correct_pred + incorrect_pred
    hosp_accuracy = correct_hosp / float (correct_hosp + incorrect_hosp)
    doc_accuracy =  correct_doc / float( correct_doc + incorrect_doc)
    pat_accuracy =  correct_pat/ float ( correct_pat + incorrect_pat)
    date_accuracy = correct_date / float ( correct_date + incorrect_date)
    phone_accuracy = correct_phone / float (correct_phone + incorrect_phone)
    ID_accuracy = correct_ID / float (correct_ID + incorrect_ID)
    overall_accuracy = correct_pred / float(total)
    return overall_accuracy,hosp_accuracy,doc_accuracy,pat_accuracy ,date_accuracy, phone_accuracy, ID_accuracy
    
          


         
        

