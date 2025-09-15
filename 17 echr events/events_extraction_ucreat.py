#!/usr/bin/env python
# coding: utf-8

import string
from tqdm import tqdm
import spacy
import os
import pandas as pd
import re
import time

# Load spaCy model
spacy.require_gpu()
nlp = spacy.load("en_core_web_trf")

# Exclusion list setup
alphabet_string = string.ascii_lowercase
alphabet_list = list(alphabet_string)
exclusion_list = alphabet_list + [
    "no", "nos", "sub-s", "subs", "ss", "cl", "dr", "mr", "mrs", "dr", "vs", "ch", "addl",
]
exclusion_list = [word + "." for word in exclusion_list]

# Preprocessing function (unchanged)
def preprocess(content):
    raw_text = re.sub(r"\xa0", " ", content)
    raw_text = raw_text.split("\n")
    text = raw_text.copy()
    text = [re.sub(r'[^a-zA-Z0-9.,<>)\-(/?\t ]', '', sentence) for sentence in text]
    text = [re.sub("\t+", " ", sentence) for sentence in text]
    text = [re.sub("\s+", " ", sentence) for sentence in text]
    text = [re.sub(" +", " ", sentence) for sentence in text]
    text = [re.sub("\.\.+", "", sentence) for sentence in text]
    text = [re.sub("\A ?", "", sentence) for sentence in text]
    text = [sentence for sentence in text if(len(sentence) != 1 and not re.fullmatch("(\d|\d\d|\d\d\d)", sentence))]
    text = [sentence for sentence in text if len(sentence) != 0]
    text = [re.sub('\A\(?(\d|\d\d\d|\d\d|[a-zA-Z])(\.|\))\s?(?=[A-Z])', '\n', sentence) for sentence in text]
    text = [re.sub("\A\(([ivx]+)\)\s?(?=[a-zA-Z0-9])", '\n', sentence) for sentence in text]
    text = [re.sub(r"[()[\]\"$']", " ", sentence) for sentence in text]
    text = [re.sub(r" no.", " number ", sentence, flags=re.I) for sentence in text]
    text = [re.sub(r" nos.", " numbers ", sentence, flags=re.I) for sentence in text]
    text = [re.sub(r" co.", " company ", sentence) for sentence in text]
    text = [re.sub(r" ltd.", " limited ", sentence, flags=re.I) for sentence in text]
    text = [re.sub(r" pvt.", " private ", sentence, flags=re.I) for sentence in text]
    text = [re.sub(r" vs\.?", " versus ", sentence, flags=re.I) for sentence in text]
    text = [re.sub(r"ors\.?", "others", sentence, flags=re.I) for sentence in text]
    
    text2 = []
    for index in range(len(text)):
        if(index > 0 and text[index] == '' and text[index-1] == ''):
            continue
        if(index < len(text)-1 and text[index+1] != '' and text[index+1][0] == '\n' and text[index] == ''):
            continue
        text2.append(text[index])
    text = text2
    text = "\n".join(text)
    lines = text.split("\n")
    text_new = " ".join(lines)
    text_new = re.sub(" +", " ", text_new)
    l_new = []
    for token in text_new.split():
        if token.lower() not in exclusion_list:
            l_new.append(token.strip())
    return " ".join(l_new)

# Dependency relation constants
SUBJECTS = ["nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl"]
OBJECTS = ["dobj", "dative", "attr", "oprd", "pobj"]
ADJECTIVES = ["acomp", "advcl", "advmod", "amod", "appos", "nn", "nmod", "ccomp", "complm", "hmod", "infmod", "xcomp", "rcmod", "poss", " possessive"]
ADVERBS = ["advmod"]
COMPOUNDS = ["compound"]
PREPOSITIONS = ["prep"]

# All helper functions (unchanged)
def getSubsFromConjunctions(subs):
    moreSubs = []
    for sub in subs:
        rights = list(sub.rights)
        rightDeps = {tok.lower_ for tok in rights}
        if "and" in rightDeps:
            moreSubs.extend([tok for tok in rights if tok.dep_ in SUBJECTS or tok.pos_ == "NOUN"])
            if len(moreSubs) > 0:
                moreSubs.extend(getSubsFromConjunctions(moreSubs))
    return moreSubs

def getObjsFromConjunctions(objs):
    moreObjs = []
    for obj in objs:
        rights = list(obj.rights)
        rightDeps = {tok.lower_ for tok in rights}
        if "and" in rightDeps:
            moreObjs.extend([tok for tok in rights if tok.dep_ in OBJECTS or tok.pos_ == "NOUN"])
            if len(moreObjs) > 0:
                moreObjs.extend(getObjsFromConjunctions(moreObjs))
    return moreObjs

def getVerbsFromConjunctions(verbs):
    moreVerbs = []
    for verb in verbs:
        rightDeps = {tok.lower_ for tok in verb.rights}
        if "and" in rightDeps:
            moreVerbs.extend([tok for tok in verb.rights if tok.pos_ == "VERB"])
            if len(moreVerbs) > 0:
                moreVerbs.extend(getVerbsFromConjunctions(moreVerbs))
    return moreVerbs

def findSubs(tok):
    head = tok.head
    while head.pos_ != "VERB" and head.pos_ != "NOUN" and head.head != head:
        head = head.head
    if head.pos_ == "VERB":
        subs = [tok for tok in head.lefts if tok.dep_ == "SUB"]
        if len(subs) > 0:
            verbNegated = isNegated(head)
            subs.extend(getSubsFromConjunctions(subs))
            return subs, verbNegated
        elif head.head != head:
            return findSubs(head)
    elif head.pos_ == "NOUN":
        return [head], isNegated(tok)
    return [], False

def isNegated(tok):
    negations = {"no", "not", "n't", "never", "none"}
    for dep in list(tok.lefts) + list(tok.rights):
        if dep.lower_ in negations:
            return True
    return False

def find_negation(tok):
    negations = {"no", "not", "n't", "never", "none"}
    for dep in list(tok.lefts):
        if dep.lower_ in negations:
            verb = dep.lower_ + " " + tok.lemma_
            verb_id = [dep.i, tok.i]
            return verb, verb_id
    verb = tok.lemma_
    verb_id = [tok.i]
    return verb, verb_id

def getObjsFromPrepositions(deps):
    objs = []
    for dep in deps:
        if dep.pos_ == "ADP" and (dep.dep_ == "prep" or dep.dep_ == "agent"):
            for tok in dep.rights:
                if (tok.pos_ == "NOUN" and tok.dep_ in OBJECTS) or (tok.pos_ == "PRON" and tok.lower_ == "me"):
                    objs.append(tok)
                elif tok.dep_ == "pcomp":
                    for t in tok.rights:
                        if (t.pos_ == "NOUN" and t.dep_ in OBJECTS) or (t.pos_ == "PRON" and t.lower_ == "me"):
                            objs.append(t)
                else:
                    objs.extend(getObjsFromPrepositions(tok.rights))
    return objs

def getAdjectives(toks):
    toks_with_adjectives = []
    for tok in toks:
        adjs = [left for left in tok.lefts if left.dep_ in ADJECTIVES]
        adjs.append(tok)
        adjs.extend([right for right in tok.rights if tok.dep_ in ADJECTIVES])
        tok_with_adj = " ".join([adj.lower_ for adj in adjs])
        toks_with_adjectives.extend(adjs)
    return toks_with_adjectives

def getObjsFromAttrs(deps):
    for dep in deps:
        if dep.pos_ == "NOUN" and dep.dep_ == "attr":
            verbs = [tok for tok in dep.rights if tok.pos_ == "VERB"]
            if len(verbs) > 0:
                for v in verbs:
                    rights = list(v.rights)
                    objs = [tok for tok in rights if tok.dep_ in OBJECTS]
                    objs.extend(getObjsFromPrepositions(rights))
                    if len(objs) > 0:
                        return v, objs
    return None, None

def getObjFromXComp(deps):
    for dep in deps:
        if dep.pos_ == "VERB" and dep.dep_ == "xcomp":
            v = dep
            rights = list(v.rights)
            objs = [tok for tok in rights if tok.dep_ in OBJECTS]
            objs.extend(getObjsFromPrepositions(rights))
            if len(objs) > 0:
                return v, objs
    return None, None

def getAllSubs(v):
    verbNegated = isNegated(v)
    subs = [tok for tok in v.lefts if tok.dep_ in SUBJECTS and tok.pos_ != "DET"]
    if len(subs) > 0:
        subs.extend(getSubsFromConjunctions(subs))
    else:
        foundSubs, verbNegated = findSubs(v)
        subs.extend(foundSubs)
    return subs, verbNegated

def getAllObjs(v):
    rights = list(v.rights)
    objs = [tok for tok in rights if tok.dep_ in OBJECTS]
    objs.extend(getObjsFromPrepositions(rights))
    potentialNewVerb, potentialNewObjs = getObjFromXComp(rights)
    if (potentialNewVerb is not None and potentialNewObjs is not None and len(potentialNewObjs) > 0):
        objs.extend(potentialNewObjs)
        v = potentialNewVerb
    if len(objs) > 0:
        objs.extend(getObjsFromConjunctions(objs))
    else:
        objs.extend(getObjsFromVerbConj(v))
    return v, objs

def getAllObjsWithAdjectives(v):
    rights = list(v.rights)
    objs = [tok for tok in rights if tok.dep_ in OBJECTS]
    if len(objs) == 0:
        objs = [tok for tok in rights if tok.dep_ in ADJECTIVES]
    objs.extend(getObjsFromPrepositions(rights))
    potentialNewVerb, potentialNewObjs = getObjFromXComp(rights)
    if (potentialNewVerb is not None and potentialNewObjs is not None and len(potentialNewObjs) > 0):
        objs.extend(potentialNewObjs)
        v = potentialNewVerb
    if len(objs) > 0:
        objs.extend(getObjsFromConjunctions(objs))
    else:
        objs.extend(getObjsFromVerbConj(v))
    return v, objs

def getObjsFromVerbConj(v):
    objs = []
    rights = list(v.rights)
    for right in rights:
        if right.dep_ == "conj":
            subs, verbNegated = getAllSubs(right)
            objs.extend(subs)
        else:
            objs.extend(getObjsFromVerbConj(right))
    return objs

def check_tag(compound):
    flag = False
    res = ""
    for token in compound:
        if token.ent_type_ == "PERSON":
            flag = True
            res = "<NAME>"
            break
        elif token.ent_type_ == "ORG":
            flag = True
            res = "<ORG>"
            break
    return flag, res

def generate_compound(token):
    token_compunds = []
    for tok in token.lefts:
        if tok.dep_ in COMPOUNDS:
            token_compunds.extend(generate_compound(tok))
    token_compunds.append(token)
    for tok in token.rights:
        if tok.dep_ in COMPOUNDS:
            token_compunds.extend(generate_compound(tok))
    return token_compunds

def generate_verb_advmod(v):
    v_compunds = []
    for tok in v.lefts:
        if tok.dep_ in ADVERBS:
            v_compunds.extend(generate_verb_advmod(tok))
    v_compunds.append(v)
    for tok in v.rights:
        if tok.dep_ in ADVERBS:
            v_compunds.extend(generate_verb_advmod(tok))
    return v_compunds

def generate_left_right_adjectives(obj):
    obj_desc_tokens = []
    for tok in obj.lefts:
        if tok.dep_ in ADJECTIVES:
            obj_desc_tokens.extend(generate_left_right_adjectives(tok))
    obj_desc_tokens.append(obj)
    for tok in obj.rights:
        if tok.dep_ in ADJECTIVES:
            obj_desc_tokens.extend(generate_left_right_adjectives(tok))
    return obj_desc_tokens

def findSVOs(tokens, len_doc):
    svos = []
    svo_token_ids = []
    verbs = [tok for tok in tokens if tok.pos_ == "VERB" and tok.dep_ != "aux"]
    for v in verbs:
        subs, verbNegated = getAllSubs(v)
        verb, verb_id = find_negation(v)
        if len(subs) > 0:
            v, objs = getAllObjs(v)
            for sub in subs:
                for obj in objs:
                    sub_compound = generate_compound(sub)
                    obj_compound = generate_compound(obj)
                    sub_flag, sub_tag = check_tag(sub_compound)
                    obj_flag, obj_tag = check_tag(obj_compound)
                    if obj_flag and sub_flag:
                        event = (sub_tag, verb, obj_tag)
                    elif obj_flag:
                        event = (" ".join(tok.lemma_ for tok in sub_compound), verb, obj_tag)
                    elif sub_flag:
                        event = (sub_tag, verb, " ".join(tok.lemma_ for tok in obj_compound))
                    else:
                        event = (" ".join(tok.lemma_ for tok in sub_compound), verb, " ".join(tok.lemma_ for tok in obj_compound))
                    svos.append(event)
    return svos, svo_token_ids

single_words = ["a", "A", "<", ">", "i", "I"]

def remove_special_characters(text):
    regex = re.compile("[^a-zA-Z<>.\s]")
    text_returned = re.sub(regex, " ", text)
    tokens = text_returned.split()
    words = []
    for word in tokens:
        if len(word) > 1 or word in single_words:
            words.append(word)
    out = " ".join(words)
    return " ".join(words)

# Updated events extraction function for single text processing
def extract_events_from_text(content):
    content = preprocess(content)
    # Define the pattern for sentence splitting
    pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s'
    # Split the text into sentences using the pattern
    content_sents = re.split(pattern, content)
    
    file_svo_text = []
    len_doc = 0
    lines = []
    
    for i, line in enumerate(content_sents):
        line = line.strip()
        lines.append(remove_special_characters(line))
    
    for i, doc in enumerate(nlp.pipe(lines)):
        SVO, SVO_Token_IDs = findSVOs(doc, len_doc)
        if len(SVO) > 0:
            for eve in SVO:
                file_svo_text.append(" ".join(eve))
    
    return file_svo_text

# CSV processing functions
def process_csv_with_events(input_csv_path, output_dir):
    """Process a single CSV file and extract events"""
    print(f"Processing: {input_csv_path}")
    df = pd.read_csv(input_csv_path)
    
    # Apply event extraction to the 'text' column
    tqdm.pandas(desc="Extracting events")
    df['events'] = df['text'].progress_apply(extract_events_from_text)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output file name based on input file name
    base_filename = os.path.splitext(os.path.basename(input_csv_path))[0]
    output_csv_path = os.path.join(output_dir, f'{base_filename}_with_events.csv')
    
    df.to_csv(output_csv_path, index=False)
    print(f"Processed file saved at: {output_csv_path}")

def process_all_csvs_in_folder(input_folder, output_folder):
    """Process all CSV files in a folder"""
    csv_files = [f for f in os.listdir(input_folder) if f.endswith(".csv")]
    print(f"Found {len(csv_files)} CSV files to process")
    
    for filename in tqdm(csv_files, desc="Processing CSV files"):
        input_csv_path = os.path.join(input_folder, filename)
        try:
            process_csv_with_events(input_csv_path, output_folder)
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

# Main execution
if __name__ == "__main__":
    # Set your input and output folders here
    input_folder = '../11 echr/prem_vs_conc_vs_NA_testing_on_echr/data'  # Folder containing CSV files to process
    output_folder = 'p_c_na_ucreat_events_echr'  # Folder to save processed CSV files
    
    # Process all CSV files
    process_all_csvs_in_folder(input_folder, output_folder)
    
    print("Event extraction completed!")
