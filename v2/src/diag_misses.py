import pickle, sys, pandas as pd
sys.path.insert(0, ".")
from normalize import normalize_name, normalize_address
from io_utils import read_source, read_ground_truth, ground_truth_to_dict

D = r"..\..\dataset\train"
cands = pd.read_pickle(r"..\..\cache\candidates_train.pkl")
gt = ground_truth_to_dict(read_ground_truth(D + r"\train_ground_truth.tsv"))
s1 = read_source(D + r"\train_source1.tsv")
pool = pd.concat([read_source(D + r"\train_source2.tsv"),
                  read_source(D + r"\train_source3.tsv")], ignore_index=True)

blocked = cands.groupby("source1_entity_id")["candidate_entity_id"].apply(set).to_dict()
S1 = {r.entity_id: r for r in s1.itertuples()}
P  = {r.entity_id: r for r in pool.itertuples()}

shares_name = shares_addr = shares_none = total = 0
for s, cs in blocked.items():
    for t in gt.get(s, []):
        if t in cs or t not in P: continue
        total += 1
        a, b = S1[s], P[t]
        n1, n2 = set(normalize_name(a.business_name).split()), set(normalize_name(b.business_name).split())
        a1, a2 = set(normalize_address(a.business_address, a.country).split()), set(normalize_address(b.business_address, b.country).split())
        if n1 & n2: shares_name += 1
        if a1 & a2: shares_addr += 1
        if not (n1 & n2) and not (a1 & a2): shares_none += 1

print(f"missed true matches: {total}")
print(f"  share a name token:    {shares_name} ({shares_name/total:.1%})")
print(f"  share an address token:{shares_addr} ({shares_addr/total:.1%})")
print(f"  share NOTHING:         {shares_none} ({shares_none/total:.1%})")