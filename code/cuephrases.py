#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
replicating Hachey and Grover's cue phrases feature set


H&G: 
    
The term ‘cue phrase’ covers the kinds of stock phrases which are frequently 
good indicators of rhetorical status (e.g. phrases such as The aim of this study
in the scientific article domain and It seems to me that in the HOLJ domain).
 Teufel and Moens invested a considerable amount of effort in building hand-crafted 
 lexicons where the cue phrases are assigned to one of a number of fixed categories. 
 A primary aim of the current research is to investigate whether this information 
 can be encoded using automatically computable linguistic features. If they can,
 then this helps to relieve the burden involved in porting systems such as these to new domains. 
 Our preliminary cue phrase feature set includes syntactic features of the main verb (voice, 
                                                                                    tense, aspect, modality, negation), 
 which we have shown in previous work to be correlated with rhetorical status (Grover et al. 2003). 
 We also use sentence initial part- of-speech and sentence initial word features to roughly 
 approximate for- mulaic expressions which are sentence-level adverbial or prepositional phrases. 
 Subject features include the head lemma, entity type, and entity subtype. 
 These features approximate the hand-coded agent features of Teufel and Moens. 
 A main verb lemma feature simulates Teufel and Moens’s type of action and a feature 
 encoding the part-of-speech after the main verb is meant to capture basic subcategorisation 
 information.
 
 [step 1]:
 
 * VOICE
 * TENSE
 * ASPECT
 * MODALITY 
 * NEGATION 
 
 [step 2]: 
 sentence Part of Speech& sentence initial word features 
     approximate formulaic expressions - sentence-level adverbial or prepositional phrases
     
     verb subject - 
         head lemma
         entity type
         entity subtype 

For each sentence,
we use part-of-speech-based heuristics to determine tense, 
voice, and presence of modal auxiliaries. This algorithm is shared 
with the metadiscourse features, and the details are described below.


THE NEW CUE PHRASES FEATURE WILL INCLUDE: 
    position (PUNCT, DET, ADJ, NOUN, VERB, DET, ADP, CCONJ, PROPN, PRON, PART) - count per sentence
    tag (JJ, NN, VBN, IN, DT, JJ, CC, VBZ, RP, PRP, LS, -LRB-, -RRB-, NNP, VB, NNS, PRP$, CD, MD) - count per sentence
    head verb (first verb that appears)
    
new cue phrases feature set ->
    
    count of dep = aux (per sentence)
    count of pos = AUX (per sentence)
    boolean of dep = aux (true or false per sentence)
    boolean of pos= AUX (true or false per sentence)

@author: amyconroy
"""

import csv
import numpy as np
import spacy

# this will be on a sentence by sentence basis that it is parsed and cut up

def cuePhrases():
    import spacy 

    nlp = spacy.load("en_core_web_sm")
    
    y = 0
    
    with open('data/UKHL_corpus.csv', 'r') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            y += 1 # keep count for writing to the file later
            
            firstVerb = False 
    
            text = row['text']
            doc = nlp(text)
            
            modal(doc, nlp)
            
            
            
            
def modal(doc, nlp):
    modalPosBool = 0 
    modalDepBool = 0
    modalDepCount = 0
    modalPosCount = 0
    

    for token in doc:
        if token.pos_ == "AUX":
            modalPosBool = 1 # true if found
            modalPosCount += 1
            
        if token.dep_ == "aux":
            modalDepCount += 1
            modalDepBool = 1 # true if found
        print(nlp.vocab.morphology.tag_map[token.tag_])
    print(modalPosBool)
    print(modalPosCount)
    print(modalDepCount)
    print(modalDepBool)
    print("\n")
    
                    
cuePhrases() 
    