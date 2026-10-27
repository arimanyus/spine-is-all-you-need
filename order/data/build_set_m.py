import json, random, collections, ast
from datasets import load_dataset
random.seed(1337)
d=load_dataset('lmsys/mt_bench_human_judgments')['human']
g=collections.defaultdict(list); conv={}
for r in d:
    if str(r['turn'])!='1': continue
    a,b=r['model_a'],r['model_b']; w=r['winner']; key=(r['question_id'],)+tuple(sorted([a,b]))
    ca,cb=r['conversation_a'],r['conversation_b']
    if (a,b)!=key[1:]:
        w={'model_a':'model_b','model_b':'model_a'}.get(w,w); ca,cb=cb,ca
    g[key].append(w); conv[key]=(ca,cb)
def ans(c):
    c=c if isinstance(c,list) else ast.literal_eval(c)
    return c[0]['content'], c[1]['content']
strata={'clear':[], 'slight':[], 'tie':[]}
for k in sorted(g):
    ws=g[k]; q,a1=ans(conv[k][0]); _,a2=ans(conv[k][1])
    if len(a1)>1500 or len(a2)>1500: continue
    top,n=collections.Counter(ws).most_common(1)[0]
    if top.startswith('tie') and n>len(ws)/2: s='tie'
    elif top.startswith('tie') or n<=len(ws)/2: continue
    elif n==len(ws) and len(ws)>=2: s='clear'
    else: s='slight'
    strata[s].append((k,q,a1,a2,{'model_a':1,'model_b':2}.get(top,'tie')))
rows=[]
for s,n in {'clear':27,'slight':27,'tie':26}.items():
    for i,(k,q,a1,a2,hw) in enumerate(random.sample(strata[s],n)):
        rows.append(dict(id=f"m-{s}-{i+1:03d}",set="M",stratum=s,question=q,answer1=a1,answer2=a2,model1=k[1],model2=k[2],human_winner=hw,question_id=k[0],possibly_memorized=True))
open("set_m.jsonl","w").write("".join(json.dumps(x)+"\n" for x in rows))
print(len(rows), collections.Counter(x['stratum'] for x in rows))
