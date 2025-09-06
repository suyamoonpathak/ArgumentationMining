# build_event_graphs.py
import os, ast, xml.etree.ElementTree as ET
from pathlib import Path

# ----------  configurable paths  ----------
XML_DIR   = Path("../3 GNN/xml_files")
CSV_DIR   = Path("updated_events/prem_vs_conc_with_events_updated")
MODEL_DIR = Path("../3 GNN/RoBERTa_prem_conc_finetuned")
OUTPUT_DIR = Path("RoBERTa_raw_graph_data_updated_event_only")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------  loader for the fine-tuned model  ----------
from transformers import AutoTokenizer, AutoModel
import torch, tqdm
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model     = AutoModel.from_pretrained(MODEL_DIR)
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

# ----------  helpers  ----------
def load_event_csv(csv_path):
    """returns dict{text -> [event, …]}"""
    import pandas as pd
    df = pd.read_csv(csv_path)
    mapping = {}
    for _, row in df.iterrows():
        try:
            events = ast.literal_eval(row["events"])
        except Exception:
            events = []
        mapping[row["text"].strip()] = events
    return mapping

def xml_nodes_edges(xml_path):
    """returns list(nodes), dict(edges), id->idx"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    nodes, seen = [], set()
    for el in root.iter():
        if el.tag not in ("prem", "conc"): continue
        nid = el.attrib.get("ID", "").strip()
        if not nid or nid in seen: continue
        nodes.append({"id": nid,
                      "text": (el.text or "").strip(),
                      "type": el.tag})
        seen.add(nid)

    id2idx = {n["id"]: i for i, n in enumerate(nodes)}
    edges = {"support": [], "attack": []}

    def split(val): return [t.strip() for t in val.split("|") if t.strip()]

    for el in root.iter():
        if el.tag not in ("prem", "conc"): continue
        src = el.attrib.get("ID", "").strip()
        if src not in id2idx: continue
        for key, ekey in (("SUP", "support"), ("ATT", "attack")):
            if key in el.attrib:
                for tgt in split(el.attrib[key]):
                    if tgt in id2idx:
                        edges[ekey].append((src, tgt,
                                            id2idx[src], id2idx[tgt]))
    return nodes, edges, id2idx

def join_events(ev_list):
    """turn ['a b', 'c d'] -> 'a b ; c d'"""
    return " ; ".join(e.strip() for e in ev_list if e.strip())

#processed embeddings
# def embed_texts(texts, batch=4):
#     embs = []
#     for i in tqdm.tqdm(range(0, len(texts), batch),
#                        desc="embedding", leave=False):
#         tok = tokenizer(texts[i:i+batch],
#                         padding=True, truncation=True,
#                         max_length=512, return_tensors="pt").to(device)
#         with torch.no_grad():
#             out = model(**tok).last_hidden_state[:,0,:].cpu()
#         embs.append(out)
#     return torch.cat(embs, 0)

#raw
def embed_texts(texts, batch=4):
    embs = []
    for i in tqdm.tqdm(range(0, len(texts), batch),
                       desc="embedding", leave=False):
        tok = tokenizer(texts[i:i+batch],
                        padding=True, truncation=True,
                        max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            token_embeddings = model.embeddings.word_embeddings(tok['input_ids'])
            cls_embeddings = token_embeddings[:,0,:]
        embs.append(cls_embeddings)
    return torch.cat(embs, 0)

def strategic_negatives(nodes, edges, id2idx,
                         support_ratio=1.0, attack_ratio=5.0):
    exist = {(id2idx[s], id2idx[t]) for typ in edges.values() for s,t,_,_ in typ}
    sup, att = len(edges["support"]), len(edges["attack"])
    target = int(sup*support_ratio + att*attack_ratio)

    node_types = ["prem" if n["type"]=="prem" else "conc" for n in nodes]
    cand = []
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i==j or (i,j) in exist: continue
            prio = 2 if (node_types[i]=="prem" and node_types[j]=="conc") else 1
            cand.append((prio,i,j))
    cand.sort(key=lambda x:-x[0])
    return [(i,j) for _,i,j in cand[:min(target,len(cand))]]

# ----------  main loop  ----------
import torch
from torch_geometric.data import Data

xml_files = sorted([p for p in XML_DIR.glob("*.xml") if p.is_file()])
print("Found", len(xml_files), "XML files")

for xml_path in tqdm.tqdm(xml_files, desc="processing"):
    csv_path = CSV_DIR / f"{xml_path.stem}_with_events.csv"
    if not csv_path.exists():
        print("  ⚠️  CSV not found for", xml_path.name, "→ skipping")
        continue
    event_map = load_event_csv(csv_path)

    # nodes / edges ---------------------------------
    nodes, edges, id2idx = xml_nodes_edges(xml_path)

    # build “event strings” -------------------------
    event_texts = []
    for n in nodes:
        ev = event_map.get(n["text"], [])
        n["ev_string"] = join_events(ev) if ev else n["text"]
        event_texts.append(n["ev_string"])

    # embeddings -----------------------------------
    emb = embed_texts(event_texts)               # (N,768)
    type_feat = torch.zeros((len(nodes),2))
    for i,n in enumerate(nodes):
        type_feat[i, 0 if n["type"]=="prem" else 1] = 1
    x = torch.cat([emb, type_feat], 1)

    # edges  ---------------------------------------
    ei, et = [], []
    for typ, code in (("support",0), ("attack",1)):
        for s,t,si,ti in edges[typ]:
            ei.append([si,ti]); et.append(code)

    for si,ti in strategic_negatives(nodes, edges, id2idx):
        ei.append([si,ti]); et.append(2)

    if not ei:          # single-node judgement
        ei = torch.empty((2,0), dtype=torch.long)
        et = torch.empty((0,),   dtype=torch.long)
    else:
        ei = torch.tensor(ei).t().contiguous()
        et = torch.tensor(et)

    y = torch.tensor([0 if n["type"]=="prem" else 1 for n in nodes])

    data = Data(
        x=x, edge_index=ei, edge_type=et, y=y,
        xml_file=xml_path.name
    )
    torch.save(data, OUTPUT_DIR / f"{xml_path.stem}.pt")

print("Graphs written to", OUTPUT_DIR)
