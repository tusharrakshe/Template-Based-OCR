from spellchecker import SpellChecker
spell = SpellChecker()
import spacy
import nltk
#python -m spacy download en_core_web_sm
import en_core_web_sm
nlp = en_core_web_sm.load()
import re

def MisSpell_Count(file_name):
    Text_file = file_name
    with open(Text_file, mode='r') as file:
        Text =  file.read()        
    Percentage_Extraction = round(sum([len(word) for word in nltk.word_tokenize(Text)])/len(Text),2)
    text = re.sub('[^A-Za-z0-9]+', ' ', Text)
#    misspelled = spell.unknown(preprocessing(Text_Cleansing(text)))
    misspelled = list(spell.unknown(nltk.word_tokenize(text)))
    ENTITIES = nlp(text)
    ALL_ENTITIES = ' '.join([str(x).lower() for x in ENTITIES.ents])
    Count = len([word for word in misspelled if word not in ALL_ENTITIES])
    Total_Words = len(nltk.word_tokenize(text))
    Error_Per = round((Count/Total_Words)*100,2)
    Error_Per = 100 - Error_Per
#    print('misspelled Count:' ,Count)
#    print('Total_Words Count:' ,Total_Words)
#    print('Total_Words Count:' ,Percentage_Extraction)
#    print('Error_Perecentage:' ,Error_Per)
    return Percentage_Extraction*100,Error_Per