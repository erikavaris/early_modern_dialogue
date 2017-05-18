#import all the things
import re
import xmltodict
import collections
import glob
from nltk.tokenize import word_tokenize

NOT_LOWERCASE = re.compile(r'^[^a-z]+$')

def get_pairs(data):
    pairs = []
    for i, item in enumerate(data):
        if i != len(data)-1:
            pair = (item, data[i+1])
            pairs.append(pair)
    return pairs

def read_shakespeare(filename):
    '''norvig.com/ngrams/shakespeare.txt
    '''
    data = []
    with open(filename, 'r') as f:
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
    foldername = '2507/2507/CEDPlain'
    dialogue_pairs = []
    #comedy dramas
    for filename in glob.glob(foldername+'/D?C*'):
        print(filename)
        dialogue = read_ced_txt(filename)
        pairs = get_pairs(dialogue)
        dialogue_pairs.extend(pairs)

    #trials
    for filename in glob.glob(foldername+'/D?T*'):
        print(filename)
        dialogue = read_ced_txt(filename)
        pairs = get_pairs(dialogue)
        dialogue_pairs.extend(pairs)

    #didactics
    for filename in glob.glob(foldername+'/D?H*'):
        print(filename)
        dialogue = read_ced_txt(filename)
        pairs = get_pairs(dialogue)
        dialogue_pairs.extend(pairs)

    #miscellaneous
    for filename in glob.glob(foldername + '/D?M*'):
        print(filename)
        dialogue = read_ced_txt(filename)
        pairs = get_pairs(dialogue)
        dialogue_pairs.extend(pairs)

    return dialogue_pairs

#quick wordcount
def wordcount(dialogue_pairs):
    words = set()
    words_l = []
    for pair in dialogue_pairs:
        for utterance in pair:
            tokens = word_tokenize(utterance)
            words.update(tokens)
            words_l.extend(tokens)

    print('unique words:', len(words))
    print('total word count:', len(words_l))




