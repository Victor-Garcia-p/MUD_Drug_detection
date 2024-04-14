#! /usr/bin/python3

import sys
import re
from os import listdir

from xml.dom.minidom import parse
from nltk.tokenize import word_tokenize
from jellyfish import jaro_winkler_similarity, levenshtein_distance
from bisect import bisect
from nltk.corpus import wordnet as wn

#import nltk
#nltk.download('wordnet')

## --------- tokenize sentence ----------- 
## -- Tokenize sentence, returning tokens and span offsets

def tokenize(txt):
    offset = 0
    tks = []
    ## word_tokenize splits words, taking into account punctuations, numbers, etc.
    for t in word_tokenize(txt):
        ## keep track of the position where each token should appear, and
        ## store that information with the token
        offset = txt.find(t, offset)
        tks.append((t, offset, offset+len(t)-1))
        offset += len(t)

    ## tks is a list of triples (word,start,end)
    return tks


## --------- get tag ----------- 
##  Find out whether given token is marked as part of an entity in the XML

def get_tag(token, spans) :
   (form,start,end) = token
   for (spanS,spanE,spanT) in spans :
      if start==spanS and end<=spanE : return "B-"+spanT
      elif start>=spanS and end<=spanE : return "I-"+spanT

   return "O"

## --------- Feature extractor ----------- 
## -- Extract features for each token in given sentence


def extract_features(tokens) :

   # for each token, generate list of features and add it to the result
   result = []

   """
   # Extract all drugs from the databases
   all_drugs = open("resources/DrugBank.txt", encoding='utf-8').readlines()
   all_drugs = [all_drugs[i].split("|")[0] for i in range(len(all_drugs))]
   all_drugs.extend(open("resources/HSDB.txt").readlines())
   all_drugs.sort()
   """

   for k in range(0,len(tokens)):
      tokenFeatures = [];
      t = tokens[k][0]
      
      # Features of the current token
      tokenFeatures.append("form="+t)
      tokenFeatures.append("suf3="+t[-3:])
      tokenFeatures.append("len=" + str(len(t)))  

      # Prefixes and suffixes for lengths 1 to 3（not only 3）
      for length in range(1, 4):
         if len(t) >= length:
            tokenFeatures.append(f"prefix{length}=" + t[:length])
            tokenFeatures.append(f"suffix{length}=" + t[-length:])
         else:
            # If the token is shorter than the length, use the whole token
            tokenFeatures.append(f"prefix{length}=" + t)
            tokenFeatures.append(f"suffix{length}=" + t)

      # Features considering the previous token
      if k>0 :
         tPrev = tokens[k-1][0]
         tokenFeatures.append("formPrev="+tPrev)
         tokenFeatures.append("suf3Prev="+tPrev[-3:])
         tokenFeatures.append("lenPrev=" + str(len(tPrev)))  # Length of the previous token
      else :
         tokenFeatures.append("BoS")

      # Features considering the next token
      if k<len(tokens)-1 :
         tNext = tokens[k+1][0]
         tokenFeatures.append("formNext="+tNext)
         tokenFeatures.append("suf3Next="+tNext[-3:])
         tokenFeatures.append("lenNext=" + str(len(tNext)))  # Length of the next token
      else:
         tokenFeatures.append("EoS")

      ## Capital letter features 
      # Binary: Is it capitalized or not
      if t[0].isupper():
         tokenFeatures.append("capitalized=Yes")
      else:
         tokenFeatures.append("capitalized=No")

      """
      # Binary: has numbers or not
      n_digits = sum(1 for c in t if c.isdigit())

      if n_digits > 0:
         tokenFeatures.append("has_numbers=Yes")
      else:
         tokenFeatures.append("has_numbers=No")
      """

      # Binary: has punctuation or not
      punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
      n_punct = sum(1 for c in t if c in punc)
      
      if n_punct > 0:
         tokenFeatures.append("has_punct=Yes")
      else:
         tokenFeatures.append("has_punct=No")
         
      """
      # Numerical: Number of capitalized words (ID 006)
      # extracted from https://stackoverflow.com/questions/18129830/count-the-uppercase-letters-in-a-string-with-python
      n_capital = sum(1 for c in t if c.isupper())
      tokenFeatures.append("ncapitalized=" + str(n_capital)) 

      # Number of digits (ID 012)
      n_digits = sum(1 for c in t if c.isdigit())
      tokenFeatures.append("ndigits=" + str(n_digits)) 

      
      # Number of punctuations (ID 012)
      punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
      n_punct = sum(1 for c in t if c in punc)
      tokenFeatures.append("npunct=" + str(n_punct)) 
      """

      """
      ## Features considering if the token is in the database (ID 002 & 003)
      # Check for perfect matches
      if t in all_drugs == True:
         tokenFeatures.append("isDrug=1")
      else:
         tokenFeatures.append("isDrug=0")

      # Check for perfect matches but putting in lowercase words at database
      # and tokens
      """

      """
      # Check for partial match (similarity functions) (ID 004 & 005)
      if len(t) < 5:
         position = bisect(all_drugs, t)
         similarity = jaro_winkler_similarity(all_drugs[position], t)
         tokenFeatures.append("Jaro_similarity=" + str(similarity))
         #similarity = levenshtein_distance(all_drugs[position], t)
         #tokenFeatures.append("Levenshtein_similarity=" + str(similarity))
      else:
         tokenFeatures.append("Levenshtein_similarity=" + "UNKNOWN")    
      """

      ## POS tagging
      # Check the POS tag of all the synsets of the token and check if there is any noun
      # (drugs are always nouns)
      """
      synset = wn.synsets(t)
      pos_tag = [synset[i].pos() for i in range(len(synset))]
      is_name = 'n' in pos_tag
      tokenFeatures.append("is_name=" + str(is_name)) 
      """

      """
      # Write the number of synsets that are nouns
      synset = wn.synsets(t)
      pos_tag = [synset[i].pos() for i in range(len(synset))]
      
      #is_name = 'n' in pos_tag
      n_name = sum(1 for t in pos_tag if t == 'n')

      if len(pos_tag) == 0: per_name = 0 
      else: per_name = n_name / len(pos_tag)

      #tokenFeatures.append("is_name=" + str(is_name) 
      #tokenFeatures.append("n_name=" + str(n_name) 
      tokenFeatures.append("perc_name=" + str(per_name)) 
      """

      result.append(tokenFeatures)
    
   return result

## --------- MAIN PROGRAM ----------- 
## --
## -- Usage:  baseline-NER.py target-dir
## --
## -- Extracts Drug NE from all XML files in target-dir, and writes
## -- them in the output format requested by the evalution programs.
## --


# directory with files to process
datadir = sys.argv[1]

# process each file in directory
for f in listdir(datadir) :
   
   # parse XML file, obtaining a DOM tree
   tree = parse(datadir+"/"+f)
   
   # process each sentence in the file
   sentences = tree.getElementsByTagName("sentence")
   for s in sentences :
      sid = s.attributes["id"].value   # get sentence id
      spans = []
      stext = s.attributes["text"].value   # get sentence text
      entities = s.getElementsByTagName("entity")
      for e in entities :
         # for discontinuous entities, we only get the first span
         # (will not work, but there are few of them)
         (start,end) = e.attributes["charOffset"].value.split(";")[0].split("-")
         typ =  e.attributes["type"].value
         spans.append((int(start),int(end),typ))
         

      # convert the sentence to a list of tokens
      tokens = tokenize(stext)

      # TEST
      """
      #POD = pos_tag(sent)
      file = open("out_logs.txt","w") 
      
      for token in tokens:
         file.write(str(token[:][0]) + " ")
      """
         
      # extract sentence features
      features = extract_features(tokens)

      # print features in format expected by crfsuite trainer
      for i in range (0,len(tokens)) :
         # see if the token is part of an entity
         tag = get_tag(tokens[i], spans) 
         print (sid, tokens[i][0], tokens[i][1], tokens[i][2], tag, "\t".join(features[i]), sep='\t')

      # blank line to separate sentences
      print()