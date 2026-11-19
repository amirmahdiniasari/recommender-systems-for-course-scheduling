import sys, numpy as np, pandas as pd, json
sys.path.insert(0,'.')
import evaluate_rq1 as ev, survey_parser as sp, recommender as rc
tidy=sp.load_survey('form_responses_23.csv')
courses=[c for c in sp.SPECIALIZED_COURSES if c in tidy.columns]
R=tidy[courses].astype(float); I=tidy['علاقه‌مندی‌ها']; Y=tidy['سال_ورود']; ids=list(R.index)
tests=ev.make_tests(R,I,Y,courses,'entry')
def exp_metrics(sc,held):
    v=sc[held]; hi=sum(1 for c,x in sc.items() if x>v+1e-12); tie=sum(1 for c,x in sc.items() if abs(x-v)<=1e-12)
    ms=[ev.metrics_from_rank(r,len(sc)) for r in range(hi+1,hi+tie+1)]
    return {k:np.mean([m[k] for m in ms]) for k in ms[0]}, tie
det={m:[] for m in ['CB','CF','Hybrid','Chart']}; exp={m:[] for m in det}; ties=[]; cbdistinct=[]; heldtie=0
fb_all=[]; fb_held=[]; nneigh=[]; commons=[]; neg_dropped=[]; small=[]
for t in tests:
    others=[x for x in ids if x!=t['u']]; train=R.loc[others]
    sc=ev.score_all(t,train,I,courses)
    for m in det:
        r=rc.rank(sc[m],courses).index(t['held'])+1
        det[m].append((t['u'],ev.metrics_from_rank(r,len(t['cands']))))
        e,tie=exp_metrics(sc[m],t['held']); exp[m].append((t['u'],e))
        if m=='CB': ties.append(tie); cbdistinct.append(len(set(round(x,9) for x in sc['CB'].values())))
    # CF diagnostics
    R2=pd.concat([train,t['ru'].to_frame(name=t['u']).T]); u=t['u']
    means=R2.mean(axis=1); mu=means[u]; ru=R2.loc[u]; sims={}; allpos=0; allneg=0
    for v in R2.index:
        if v==u: continue
        common=[c for c in R2.columns if not pd.isna(ru[c]) and not pd.isna(R2.loc[v,c])]
        if len(common)<2: continue
        a=np.array([ru[c]-mu for c in common]); b=np.array([R2.loc[v,c]-means[v] for c in common]); s=rc._cos(a,b)
        if s>0: sims[v]=(s,len(common)); allpos+=1
        else: allneg+=1
    top=sorted(sims.items(),key=lambda x:-x[1][0])[:5]
    nneigh.append(len(top)); commons+= [x[1][1] for x in top]; neg_dropped.append(allneg)
    fbs=[]
    for c in t['cands']:
        has=any(not pd.isna(R2.loc[v,c]) for v,_ in top)
        fbs.append(not has)
    fb_all.append(np.mean(fbs)); fb_held.append(fbs[t['cands'].index(t['held'])])
def agg(d,k):
    per={}
    for u,m in d: per.setdefault(u,[]).append(m[k])
    return float(np.mean([np.mean(v) for v in per.values()]))
out={'ties':{'mean_group_of_held':float(np.mean(ties)),'frac_held_in_tie':float(np.mean([x>1 for x in ties])),'mean_distinct_scores':float(np.mean(cbdistinct)),'mean_cands':float(np.mean([len(t['cands']) for t in tests]))},
'cmp':{m:{'R@3_det':agg(det[m],'R@3'),'R@3_exp':agg(exp[m],'R@3'),'N5_det':agg(det[m],'NDCG@5'),'N5_exp':agg(exp[m],'NDCG@5')} for m in det},
'cf':{'frac_cands_fallback':float(np.mean(fb_all)),'frac_tests_held_fallback':float(np.mean(fb_held)),'mean_neighbors':float(np.mean(nneigh)),'frac_tests_lt5_neighbors':float(np.mean([n<5 for n in nneigh])),'mean_neg_or_zero_sim_dropped':float(np.mean(neg_dropped)),'common_mean':float(np.mean(commons)),'common_median':float(np.median(commons)),'frac_neighbors_common_le3':float(np.mean([c<=3 for c in commons])),'frac_common_eq2':float(np.mean([c==2 for c in commons]))}}
# variants: CF fallback = mu ; CF min_common sweep
orig=rc.cf_scores
def cf_fb_mu(R,u,cands,k=5):
    out=orig(R,u,cands,k); 
    means=R.mean(axis=1); mu=means[u]
    # recompute fallback as mu
    ru=R.loc[u]; sims={}
    for v in R.index:
        if v==u: continue
        common=[c for c in R.columns if not pd.isna(ru[c]) and not pd.isna(R.loc[v,c])]
        if len(common)<rc.MIN_COMMON: continue
        a=np.array([ru[c]-mu for c in common]); b=np.array([R.loc[v,c]-means[v] for c in common]); s=rc._cos(a,b)
        if s>0: sims[v]=s
    top=sorted(sims.items(),key=lambda x:-x[1])[:k]
    for c in cands:
        if not any(not pd.isna(R.loc[v,c]) for v,_ in top): out[c]=mu
    return out
def run():
    per=ev.evaluate(tests,ids,R,I,courses); return per.groupby('method')[['R@3','NDCG@5']].mean()
var={}
ev.cf_scores=cf_fb_mu; s=run(); var['fallback_mu']={m:[s.loc[m,'R@3'],s.loc[m,'NDCG@5']] for m in ['CF','Hybrid']}
ev.cf_scores=orig
for mc in [2,3,4,5]:
    rc.MIN_COMMON=mc; s=run(); var[f'minc_{mc}']={m:[s.loc[m,'R@3'],s.loc[m,'NDCG@5']] for m in ['CF','Hybrid']}
rc.MIN_COMMON=2
out['var']=var
json.dump(out,open('./diag.json','w'),ensure_ascii=False,indent=1); print(json.dumps(out,ensure_ascii=False,indent=1))
