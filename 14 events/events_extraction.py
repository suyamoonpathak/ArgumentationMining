# first activate the events conda environment

import pandas as pd
import os
import spacy
import re
import string

# Load your spaCy model
nlp = spacy.load("en_core_web_trf")  # Or use 'en_core_web_sm' if transformer not installed

# Exclusion list for preprocessing
alphabet_string = string.ascii_lowercase
alphabet_list = list(alphabet_string)
exclusion_list = alphabet_list + [
    "no", "nos", "sub-s", "subs", "ss", "cl", "dr", "mr", "mrs", "dr", "vs", "ch", "addl",
]
exclusion_list = [word + "." for word in exclusion_list]

def preprocess(content):
    raw_text = re.sub(r"\xa0", " ", content)
    raw_text = raw_text.split("\n")
    text = raw_text.copy()
    text = [re.sub(r'[^a-zA-Z0-9.,<>)\/\-\t ]', r'', sentence) for sentence in text]
    text = [re.sub("\\t+", " ", sentence) for sentence in text]
    text = [re.sub("\\s+", " ", sentence) for sentence in text]
    text = [re.sub(" +", " ", sentence) for sentence in text]
    text = [re.sub("\\.\\.+", "", sentence) for sentence in text]
    text = [re.sub("\\A ?", "", sentence) for sentence in text]
    text = [sentence for sentence in text if (len(sentence) != 1 and not re.fullmatch("(\\d|\\d\\d|\\d\\d\\d)", sentence))]
    text = [sentence for sentence in text if len(sentence) != 0]
    text = [re.sub('\\A\\(?([\\d\\w]{1,3})(\\.|\\))\\s?(?=[A-Z])', '\\n', sentence) for sentence in text]
    text = [re.sub("\\A\\(([ivx]+)\\)\\s?(?=[a-zA-Z0-9])", '\\n', sentence) for sentence in text]
    text = [re.sub(r"[()\\[\\]\\\"$']", " ", sentence) for sentence in text]
    text = [re.sub(r" no.", " number ", sentence, flags=re.I) for sentence in text]
    text = [re.sub(r" nos.", " numbers ", sentence, flags=re.I) for sentence in text]
    text = [re.sub(r" co.", " company ", sentence) for sentence in text]
    text = [re.sub(r" ltd.", " limited ", sentence, flags=re.I) for sentence in text]
    text = [re.sub(r" pvt.", " private ", sentence, flags=re.I) for sentence in text]
    text = [re.sub(r" vs\\.?", " versus ", sentence, flags=re.I) for sentence in text]
    text = [re.sub(r"ors\\.?", "others", sentence, flags=re.I) for sentence in text]
    text2 = []
    for index in range(len(text)):
        if(index > 0 and text[index] == '' and text[index-1] == ''):
            continue
        if(index < len(text)-1 and text[index+1] != '' and text[index+1][0] == '\\n' and text[index] == ''):
            continue
        text2.append(text[index])
    text = text2
    text = "\\n".join(text)
    lines = text.split("\\n")
    text_new = " ".join(lines)
    text_new = re.sub(" +", " ", text_new)
    l_new = []
    for token in text_new.split():
        if token.lower() not in exclusion_list:
            l_new.append(token.strip())
    return " ".join(l_new)

# Event extraction helpers (minimal version as above, full pipeline can be added)
SUBJECTS = {"nsubj", "nsubjpass", "csubj", "csubjpass", "expl"}
OBJECTS  = {"dobj", "attr", "oprd", "pobj"}        # keep
PASSIVE_PREP = {"agent", "prep"}                   # for by-phrases etc.

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
            return verb
    return tok.lemma_

def inherit_subject_from_conj(v):
    """
    If v has no nsubj/nsubjpass, walk to its coordinated head
    and reuse its subject(s). Handles 'succeed ... and be rejected'.
    """
    subject = []
    if v.dep_ == "conj" and v.head != v:          # coordinate verb
        for tok in v.head.lefts:
            if tok.dep_ in SUBJECTS:
                subject.append(tok)
    return subject

def getAllSubs(v):
    subs = [tok for tok in v.lefts if tok.dep_ in SUBJECTS and tok.pos_ != "DET"]
    if not subs:
        subs = inherit_subject_from_conj(v)
    return subs


def getAllObjs(v):
    """
    1. direct objects                VERB -> dobj/attr/oprd
    2. prepositional objs            VERB -> prep/agent -> pobj
    3. allow zero-object verbs
    """
    rights = list(v.rights)
    objs   = [tok for tok in rights if tok.dep_ in OBJECTS]

    # search one hop through prep/agent
    for r in rights:
        if r.dep_ in PASSIVE_PREP:
            objs.extend(t for t in r.rights if t.dep_ == "pobj")

    return objs                           # may be []

# ------------------------------------------------------------------
def full_verb_phrase(v):
    """
    Combine auxiliaries + negation with the main verb: 'must be rejected'
    """
    auxiliaries = [tok for tok in v.lefts if tok.dep_ in {"aux", "auxpass"}]
    negs        = [tok for tok in v.lefts if tok.dep_ == "neg"]
    parts = auxiliaries + negs + [v]
    return " ".join(tok.lemma_.lower() for tok in parts)


def findSVOs(doc):
    events = []
    for v in [t for t in doc if t.pos_ == "VERB" and t.dep_ != "aux"]:
        subs = getAllSubs(v)
        objs = getAllObjs(v)
        verb_phrase = full_verb_phrase(v)

        # SV-O
        for s in subs:
            for o in objs:
                events.append((s.lemma_.lower(), verb_phrase, o.lemma_.lower()))

        # SV only  (keep even if obj missing)
        if subs and not objs:
            for s in subs:
                events.append((s.lemma_.lower(), verb_phrase, ""))

        # VO only  (rare in legal text, but keep for completeness)
        if objs and not subs:
            for o in objs:
                events.append(("", verb_phrase, o.lemma_.lower()))
    return events

def extract_events_from_text(text):
    preprocessed_text = preprocess(text)
    # Simplified pattern to split sentences based on periods and question marks.
    # This approach avoids using problematic look-behind assertions.
    pattern = r'(?<=[.?!])\s+'
    sentences = re.split(pattern, preprocessed_text)
    sentences = [s.strip() for s in sentences if s.strip() != '']
    events = []
    docs = list(nlp.pipe(sentences))
    for doc in docs:
        svos = findSVOs(doc)
        if svos:
            events.extend([" ".join(event) for event in svos])
    return events


def process_csv_with_events(input_csv_path, output_dir):
    df = pd.read_csv(input_csv_path)
    df['events'] = df['text'].apply(extract_events_from_text)
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output file name based on input file name
    base_filename = os.path.splitext(os.path.basename(input_csv_path))[0]
    output_csv_path = os.path.join(output_dir, f'{base_filename}_with_events.csv')
    
    df.to_csv(output_csv_path, index=False)
    print(f"Processed file saved at: {output_csv_path}")

# New function to process all CSVs in a folder
def process_all_csvs_in_folder(input_folder, output_folder):
    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            input_csv_path = os.path.join(input_folder, filename)
            process_csv_with_events(input_csv_path, output_folder)

# Example Usage
input_folder = '../7 Dataset/argumentative_nonargumentative_FinalCSVs/all_arg_vs_nonarg'  # Set the folder path with the CSVs
output_folder = 'arg_vs_non-arg_with_events_updated'  # Set the folder path where to save updated files

process_all_csvs_in_folder(input_folder, output_folder)
