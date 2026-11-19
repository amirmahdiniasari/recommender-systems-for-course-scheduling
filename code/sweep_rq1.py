import sys, json, numpy as np, pandas as pd
sys.path.insert(0,'.')
import evaluate_rq1 as ev, survey_parser as sp
tidy=sp.load_survey('form_responses_23.csv')
courses=[c for c in sp.SPECIALIZED_COURSES if c in tidy.columns]
R=tidy[courses].astype(float); I=tidy['علاقه‌مندی‌ها']; Y=tidy['سال_ورود']; ids=list(R.index)
tests=ev.make_tests(R,I,Y,courses,'entry')
def run(tests=tests,R=R,I=I):
    per=ev.evaluate(tests,list(R.index),R,I,courses)
    s=per.groupby('method')[['R@3','NDCG@5','MRR']].mean()
    return s
out={}
orig=ev.cf_scores
for a in [0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1]:
    ev.ALPHA=a; s=run(); out[f'alpha_{a}']={'Hybrid':[s.loc['Hybrid','R@3'],s.loc['Hybrid','NDCG@5']],'Chart':[s.loc['Chart','R@3'],s.loc['Chart','NDCG@5']]}
ev.ALPHA=.3
for k in [1,2,3,5,8,12,22]:
    ev.cf_scores=lambda R2,u,c,k=k: orig(R2,u,c,k=k); s=run(); out[f'k_{k}']={'CF':[s.loc['CF','R@3'],s.loc['CF','NDCG@5']],'Hybrid':[s.loc['Hybrid','R@3'],s.loc['Hybrid','NDCG@5']]}
ev.cf_scores=orig
for b in [0,.25,.5,1,2,4]:
    ev.BETA=b; s=run(); out[f'beta_{b}']={'Chart':[s.loc['Chart','R@3'],s.loc['Chart','NDCG@5']]}
ev.BETA=1.0
# prereq off
po=ev.prereq_ok; ev.prereq_ok=lambda *a,**k: True
t2=ev.make_tests(R,I,Y,courses,'entry'); s=run(t2); out['noprereq']={'n_tests':len(t2),'n_students':len({t['u'] for t in t2}),'cands':float(np.mean([len(t['cands']) for t in t2])),**{m:[s.loc[m,'R@3'],s.loc[m,'NDCG@5']] for m in ['Popularity','CF','Hybrid','Chart','CB']}}
ev.prereq_ok=po
# error analysis
rows=[]
for t in tests:
    others=[x for x in ids if x!=t['u']]; train=R.loc[others]
    sc=ev.score_all(t,train,I,courses)
    r={'u':t['u'],'held':t['held'],'n':len(t['cands'])}
    for m in ['Popularity','CF','Hybrid','Chart','CB']: r[m]=ev.rank(sc[m],courses).index(t['held'])+1
    rows.append(r)
E=pd.DataFrame(rows); E.to_csv('./errors.csv',index=False,encoding='utf-8-sig')
json.dump(out,open('./sweeps.json','w'),ensure_ascii=False,indent=1)
print('done',len(E))
