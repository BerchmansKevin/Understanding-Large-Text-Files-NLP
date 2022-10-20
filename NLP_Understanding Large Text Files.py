#!/usr/bin/env python
# coding: utf-8

# # BERCHMANS KEVIN S
# 
# 
# # NLP LAB

# # Understanding Large Text Files
# 
# 
# ## EXERCISE - 1
# 

# In[1]:


import nltk 
nltk.download('wordnet')
text = "This is Andrew's text, isn't it?"


# 1.How many tokens are there if you use WhitespaceTokenizer?. Print tokens.

# In[2]:


tokenizer = nltk.tokenize.WhitespaceTokenizer() 
tokens = tokenizer.tokenize(text)
print(len(tokens))
print(tokens)


# 2.How many tokens are there if you use TreebankWordTokenizer?. Print tokens.
# 

# In[3]:


tokenizer = nltk.tokenize.TreebankWordTokenizer()
tokens = tokenizer.tokenize(text)
print(tokens)


# 3.How many tokens there are if you use WordPunctTokenizer?. Print tokens.

# In[4]:


tokenizer = nltk.tokenize.WordPunctTokenizer()
tokens = tokenizer.tokenize(text)
print(tokens)


# ## EXERCISE - 2

# 1.Open the file: O. Henry's The Gift of the Magi (gift-of-magi.txt).

# In[5]:


file1 = open("E:/BHC Semester 2/NLP LAB/nlp lab1/gift-of-magi.txt", 'r')
con=file1.read()
print(con)


# 2.Write a Python script to print out the following:
# 
# 
#   1.How many word tokens there are
# 
#   

# In[6]:


nltk.download('punkt')


# In[7]:


tokenizer = nltk.tokenize.WhitespaceTokenizer() 
tokens = tokenizer.tokenize(con) 
print(len(tokens)) 


# 2.How many word types there are, (word types are a unique set of words)
# 
#  

# In[8]:


from nltk import * 
data=FreqDist(tokens)
data 


#  3.Top 20 most frequent words and their counts
# 
#   

# In[9]:


data.most_common(20) 


# 4.Words that are at least 10 characters long and their counts
# 
#   

# In[10]:


from nltk import * 
test=[w for w in tokens if len(w) >10]
freq=FreqDist(test)
freq


# 5.10+ characters-long words that occur at least twice, sorted from most frequent to least
# 

# In[11]:


for o,p in freq.items(): 
    if len(o) > 10 and p>2: 
        print(o,p) 


# ## EXERCISE - 3

# List Comprehension 
# 
# STEP-1 Download the document Austen's Emma ("austen-emma.txt"). Read it in and apply the usual text processing steps, building three objects: etoks (a list of word tokens, all in lowercase), etypes (an alphabetically sorted word type list), and efreq (word frequency distribution). 
# 

# In[12]:


fname = "austen-emma.txt" 
file1 = open(fname, 'r') 
etxt = file1.read() 
print(etxt) 
file1.close() 


# STEP 2: list-comprehend Emma Now, explore the three objects wlist, efreq, and etypes to answer the following questions. Do NOT use the for loop! Every solution must involve use of LIST COMPREHENSION. 
# 

# In[13]:


fname = "austen-emma.txt" 
f = open(fname, 'r')
etxt = f.read() 
f.close() 
etxt[-200:] 


# In[14]:


etoks = nltk.word_tokenize(etxt.lower()) 
etoks[-20:] 


# In[15]:


etypes = sorted(set(etoks)) 
etypes[-10:] 


# In[16]:


len(etypes)


# In[17]:


efreq = nltk.FreqDist(etoks) 


# Question 1
# 
# Words with prefix and suffix What are the words that start with 'un' and end in 'able'? 

# In[18]:


[word for word in etoks if word.startswith("un") & word.endswith("able")] 


# Question 2
# 
# Length How many Emma word types are 15 characters or longer? Exclude hyphenated words. 

# In[19]:


tokenizer = nltk.tokenize.WordPunctTokenizer() 
toke = tokenizer.tokenize(etxt) 


# In[20]:


[word for word in toke if len(word)>15] 


# Question 3
# 
# Average word length What's the average length of all Emma word types? 
# 

# In[21]:


average=sum(len(word) for word in toke)/len(toke) 
average 


#  Question 4: Word frequency 
#  
# How many Emma word types have a frequency count of 200 or more? How many word tyr 
# 

# In[22]:


from nltk import * 
fdiemm = FreqDist(tokens)


# In[23]:


for o,p in fdiemm.items(): 
    if p > 200: 
        print(o,p) 


# Question 5
# 
# Emma words not in wlist of the Emma word types, how many of them are not found in our list of ENABLE English words, i.e., wlist? 
# 
# STEP 3: bigrams in Emma 
# 
# 
# Let's now try out bigrams. Build two objects: e2grams (a list of word bigrams; make sure to cast it as a list) and e2gramfd (a frequency distribution of bigrams) as shown below, and then answer the following questions. 
# 
# 

# In[24]:


e2grams = list(nltk.bigrams (toke)) 
e2gramfd = nltk.FreqDist(e2grams) 
e2gramfd 


# Question 6
#     
# Bigrams What are the last 10 bigrams? 
# 

# In[25]:


last_ten = FreqDist(dict(e2gramfd.most_common( ) [-10:]))
last_ten 


# Question 7
# 
# Bigram top frequency What are the top 20 most frequent bigrams? 
# 

# In[26]:


tokenizer = nltk.tokenize.WhitespaceTokenizer() 
tokes = tokenizer.tokenize(etxt)


# In[27]:


e2grams = list(nltk.bigrams (tokes)) 
e2gramfd = nltk. FreqDist(e2grams) 


# In[28]:


e2gramfd. most_common (20) 


# Question 8
# 
# Bigram frequency count How many times does the bigram 'so happy' appear? 
# 

# In[29]:


for o,p in e2gramfd.items(): 
    if o == ('so', 'happy'): 
        print(o,p) 


# Question 9
# 
# Word following 'so' What are the words that follow 'so'? What are their frequency counts? 
# (For loop will be easier; see if you can utilize list comprehension for this.) 
# 

# In[30]:


import re 
from collections import Counter 


# In[31]:


words = re.findall(r'so+ \w+', open('austen-emma.txt').read()) 
ab = Counter(zip(words))
print(ab) 


# Question 10
# 
# Trigrams What are the last 10 trigrams? (You can use nitk.util.ngrams () method) 
# 

# In[32]:


e3grams = list(nltk.trigrams (tokes)) 
e3gramfd = nltk.FreqDist(e3grams) 


# In[33]:


last_ten = FreqDist(dict(e3gramfd.most_common ()[-10:])) 
last_ten 


# Question 11
# 
# Trigram top frequency 
# 

# In[34]:


e3gramfd.most_common(10) 


# Question 12
# 
# Trigram frequency count How many times does the trigram 'so happy to appear? 
# 

# In[35]:


for o,p in e3gramfd.items(): 
    if o == ('so', 'happy', 'to'): 
        print(o,p) 


# In[ ]:




