#import all the things
import re
import xmltodict
import collections
import glob
from nltk.tokenize import word_tokenize
import pandas as pd
#import seaborn
import json

NOT_LOWERCASE = re.compile(r'^[^a-z]+$')

def get_pairs(data):
    pairs = []
    for i, item in enumerate(data):
        if i != len(data)-1:
            pair = (item, data[i+1])
            pairs.append(pair)
    return pairs

def read_shakespeare():
    '''data/shakespeare.txt
    '''
    data = []
    with open('data/shakespeare.txt', 'r') as f:
        data = f.read()

    #divide by double newlines
    data = data.split('\n\n')

    #take out headers for names of plays etc.
    for (i, line) in enumerate(data):
        if re.match(NOT_LOWERCASE, line):
            data.pop(i)

    pairs = get_pairs(data)

    return pairs

def read_ced_xml(foldername):
    #can be used, but will delete all char names from inside dialogue
    #eg. a name referenced as object of sentence
    data = []
    for filename in glob.glob(foldername+'/*'):
        dialogue = []
        with open(filename, 'r') as f:
            xmldoc = xmltodict.parse(f.read())
            for item in xmldoc['dialogueDoc']['dialogueText']['dialogue']:
                if type(item)==collections.OrderedDict:
                    if '#text' in item.keys():
                        dialogue.append(item['#text'])
                else:
                    dialogue.append(item)
        data.extend(get_pairs(dialogue))

    return data

def read_ced_drama(filename):
    dialogue = []
    INTRO_TEXT = re.compile(r'\<.+\>', re.I)
    NOTES_2 = re.compile(r'\[\^[^\^\]\[]+\^\]', re.I) #excludes ^
    NOTES_1 = re.compile(r'\[\^[\w\s"\(\)\^\.\,\d:-]+\^\]', re.I) #covers ^

    STAGE_DIR = r'\[\$[^$\]]+\$\]'
    ANOTHER_STAGE_DIR = r'\[\}[^\}]+\}\]'

    FONT = re.compile(r'\(\^[\w \']+\^\)', re.I)
    ANOTHER_FONT = re.compile(r'\(\^ \(\\[^\^\)]+\\\) \^\)')
    PRE_FONT = re.compile(r'\(\^(?=[\w \']+\^\))', re.I)
    POST_FONT = re.compile(r'(?<=[a-zA-Z])\^\)', re.I)
    PRE_ANOTHER_FONT = re.compile(r'\(\^ \(\\(?=[^\\\)]+\\\) \^\))')
    POST_ANOTHER_FONT = re.compile(r'(?<=[a-zA-Z.])\\\) \^\)')

    DIALOGUE_SPLIT = re.compile(STAGE_DIR + '|' + ANOTHER_STAGE_DIR)


    with open(filename, 'rb') as f:
        data = f.read()
        try:
            data = data.decode('utf-8')
        except UnicodeDecodeError: #some files contain unidified encoding
            data = ''.join(map(chr, data)) #manual conversion of bytes to str
    #take out non-dialogue texts
    data = re.sub(INTRO_TEXT, '', data)
    data = re.sub(NOTES_2, '', data)
    if re.findall(NOTES_1, data):
        data = re.sub(NOTES_1, '', data)
    data = re.sub('#\n', '\n', data) #some texts have # before line breaks

    if re.findall(ANOTHER_FONT, data): #take out weird font notations around words
        data = re.sub(PRE_ANOTHER_FONT, '', data)
        data = re.sub(POST_ANOTHER_FONT, '', data)

    if re.findall(FONT, data): #smaller version of weird font notation
        data = re.sub(PRE_FONT, '', data)
        data = re.sub(POST_FONT, '', data)

    data = re.sub('   ', '', data)

    dialogue = re.split(DIALOGUE_SPLIT, data)
    clean_dialogue = []
    for i,line in enumerate(dialogue):
        line = line.rstrip()
        #line = line.rstrip('\\n')
        if line == '.':
            continue
        if line == '\n':
            continue
        if line == '':
            continue
        else:
            clean_dialogue.append(line)

    return clean_dialogue

def read_ced_txt(filename):
    #This will cover both trials and didactic works
    dialogue = []
    INTRO_TEXT = re.compile(r'\<.+\>', re.I)
    NOTES_2 = re.compile(r'\[\^[^\^\]\[]+\^\]', re.I) #excludes ^
    NOTES_1 = re.compile(r'\[\^[\w\s"\(\)\^\.\,\d:-]+\^\]', re.I) #covers ^

    STAGE_DIR = r'\[\$[^$\]]+\$\]'
    ANOTHER_STAGE_DIR = r'\[\}[^\}]+\}\]'

    FONT = re.compile(r'\(\^[\w \']+\^\)', re.I)
    ANOTHER_FONT = re.compile(r'\(\^ \(\\[^\^\)]+\\\) \^\)')
    PRE_FONT = re.compile(r'\(\^(?=[\w \']+\^\))', re.I)
    POST_FONT = re.compile(r'(?<=[a-zA-Z])\^\)', re.I)
    PRE_ANOTHER_FONT = re.compile(r'\(\^ \(\\(?=[^\\\)]+\\\) \^\))')
    POST_ANOTHER_FONT = re.compile(r'(?<=[a-zA-Z.])\\\) \^\)')

    DIALOGUE_SPLIT = re.compile(STAGE_DIR + '|' + ANOTHER_STAGE_DIR)


    with open(filename, 'rb') as f:
        data = f.read()
        try:
            data = data.decode('utf-8')
        except UnicodeDecodeError: #some files contain unidified encoding
            data = ''.join(map(chr, data)) #manual conversion of bytes to str
    #take out non-dialogue texts
    data = re.sub(INTRO_TEXT, '', data)
    data = re.sub(NOTES_2, '', data)
    if re.findall(NOTES_1, data):
        data = re.sub(NOTES_1, '', data)
    data = re.sub('#\n', '\n', data) #some texts have # before line breaks

    if re.findall(ANOTHER_FONT, data): #take out weird font notations around words
        data = re.sub(PRE_ANOTHER_FONT, '', data)
        data = re.sub(POST_ANOTHER_FONT, '', data)

    if re.findall(FONT, data): #smaller version of weird font notation
        data = re.sub(PRE_FONT, '', data)
        data = re.sub(POST_FONT, '', data)

    data = re.sub('   ', '', data)

    dialogue = re.split(DIALOGUE_SPLIT, data)
    clean_dialogue = []
    for i,line in enumerate(dialogue):
        line = line.rstrip()
        #line = line.rstrip('\\n')
        if line == '.':
            continue
        if line == '\n':
            continue
        if line == '':
            continue
        else:
            clean_dialogue.append(line)

    return clean_dialogue

def read_ced():
    '''Open up each CED file relevant to our dialogue test
    and extract the dialogues from them.
    '''
    foldername = 'data/2507/2507/CEDPlain' #adjust to wherever CED data lives
    dialogue_pairs = []
    #comedy dramas
    for filename in glob.glob(foldername+'/D?C*'):
        print(filename)
        dialogue = read_ced_txt(filename)
        #error message
        if len(dialogue) < 20:
            print('Potential Error in parsing:', filename)
        pairs = get_pairs(dialogue)
        dialogue_pairs.extend(pairs)

    #trials
    for filename in glob.glob(foldername+'/D?T*'):
        print(filename)
        dialogue = read_ced_txt(filename)
        #error message
        if len(dialogue) < 20:
            print('Potential Error in parsing:', filename)
        pairs = get_pairs(dialogue)
        dialogue_pairs.extend(pairs)

    #didactics
    for filename in glob.glob(foldername+'/D?H*'):
        #don't do D1HFDESA; it has no character separations
        if not filename.endswith('D1HFDESA'):
            print(filename)
            dialogue = read_ced_txt(filename)
            #error message
            if len(dialogue) < 20:
                print('Potential Error in parsing:', filename)
            pairs = get_pairs(dialogue)
            dialogue_pairs.extend(pairs)

    #miscellaneous
    for filename in glob.glob(foldername + '/D?M*'):
        #don't do D3HFMAUG, it's not fixed yet
        if not filename.endswith('D3HFMAUG'):
            print(filename)
            dialogue = read_ced_txt(filename)
            #error message
            if len(dialogue) < 20:
                print('Potential Error in parsing:', filename)
            pairs = get_pairs(dialogue)
            dialogue_pairs.extend(pairs)

    return dialogue_pairs

def write_datafiles(dialogue_pairs):
    with open('data/input_data.json', 'w') as wi:
        with open('data/output_data.json', 'w') as wo:
            for pair in dialogue_pairs:
                itext = {'text': pair[0]}
                otext = {'text': pair[1]}
                json.dump(itext, wi)
                wi.write('\n')
                json.dump(otext, wo)
                wo.write('\n')

#quick wordcount
def wordcount(dialogue_pairs):
    words = set()
    words_l = []
    lengths = []
    for pair in dialogue_pairs:
        l = {}
        for i, utterance in enumerate(pair):
            tokens = word_tokenize(utterance)
            l['l'+str(i)] = len(tokens)
            words.update(tokens)
            words_l.extend(tokens)
        lengths.append(l)

    print('unique words:', len(words))
    print('total word count:', len(words_l))

    return lengths

def bucket_description(lengths):

    df = pd.DataFrame(lengths)
    #column for sequence of l0 and l1
    def f(row):
        return [row['l0'], row['l1']]

    df['sequence'] = df.apply(f, axis=1)
    df['difference'] = df['l1'] - df['l0']
    #histogram & descriptive stats to determine range of most data
    df['difference'].describe()
    df['difference'].plot.hist()
    #take slice of most frequent data dist
    middle_slice = df[(df['difference']>= -14) & (df['difference']>= 14)]
    #cross tabs to see combinations of l0 and l1
    crosstabs = pd.crosstab(middle_slice['l0'], middle_slice['l1'])
    #plot crosstabs in heatmap
    f,x = plt.subplots()
    seaborn.heatmap(crosstabs)




