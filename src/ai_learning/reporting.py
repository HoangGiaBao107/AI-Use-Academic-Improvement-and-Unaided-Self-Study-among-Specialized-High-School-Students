"""Create conclusion tables, monochrome figures, and table-download bundles."""
from pathlib import Path
import html,zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .storage import save_table,save_figure,finish_phase

PHASE='07_report_outputs'
LABELS={'primary':'Primary','no_A7':'No A7','A7_Good_only':'A7 Good only','AOR_no2':'Without AOR2','AOR_no3':'Without AOR3',
        'no_straightline':'No straight-line flags','no_review_flags':'No review flags','B2_categorical':'Categorical B2',
        'expanded_constructs':'Expanded','nonlinear_scores':'Nonlinear'}

def read(root,phase,name):return pd.read_csv(Path(root)/'tables'/phase/(name+'.csv'))

def run(args):
    root=Path(args.output)
    plt.rcParams.update({'font.size':10,'font.family':'DejaVu Serif','axes.spines.top':False,'axes.spines.right':False,
                         'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white','svg.hashsalt':'ai-learning-report-v1'})
    boot=read(root,'04_uncertainty','bootstrap_intervals')
    main=read(root,'03_associations','main_models').merge(boot,on=['group','scenario','outcome'],validate='one_to_one')
    comp=read(root,'03_associations','c3_components').merge(boot,on=['group','scenario','outcome'],validate='one_to_one')
    save_table(root,PHASE,'main_results',main)
    save_table(root,PHASE,'component_results',comp)
    claims=pd.DataFrame([
        ['C1a','Supported; relatively robust','Higher CAU is associated with reported academic improvement.'],
        ['C1b','Primary support; limited robustness','The joint-improvement association is sensitive to specification.'],
        ['C2','Primary-family support; limited robustness','The divergence association weakens under several sensitivity checks.'],
        ['C3','Supported; relatively robust','Higher AOR is associated with lower probability of joint improvement.'],
        ['C4','Not established','Predictive superiority of behavior over quantity was not established.'],
        ['C5','Not established','A continuous CAU-by-AOR interaction was not established.'],
        ['C6','Descriptive','School-context responses motivate evaluated classroom proposals.']],columns=['question','report_assessment','interpretation'])
    save_table(root,PHASE,'evidence_summary',claims)

    fig,ax=plt.subplots(figsize=(8,4.6));ax.axis('off')
    def box(x,y,text,w=.27,h=.15):
        ax.text(x,y,text,ha='center',va='center',bbox=dict(boxstyle='square,pad=.6',facecolor='white',edgecolor='black'),transform=ax.transAxes)
    box(.3,.85,'Invited: 522');box(.3,.53,'Responded: 358\n68.6% of invitees');box(.3,.18,'Analyzed: 323\n90.2% of respondents')
    box(.78,.72,'No response: 164');box(.78,.36,'Not analyzed: 35\nOverall response-to-analysis difference')
    for a,b in [((.3,.77),(.3,.63)),((.3,.43),(.3,.28)),((.3,.72),(.6,.72)),((.3,.36),(.55,.36))]:
        ax.annotate('',xy=b,xytext=a,xycoords='axes fraction',arrowprops=dict(arrowstyle='->',color='black'))
    ax.set_title('Participant flow');save_figure(root,'01_participant_flow',fig);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(8,4),sharey=True)
    for ax,question,low,high,predictor,title in [(axes[0],'C1a',3,4,'CAU score','Academic improvement'),(axes[1],'C3',2,3,'AOR score','Joint improvement')]:
        r=main.loc[main.question.eq(question)].iloc[0];p=np.array([r.p_low,r.p_high])*100
        lo=np.array([r.p_low_CI95_low,r.p_high_CI95_low])*100;hi=np.array([r.p_low_CI95_high,r.p_high_CI95_high])*100
        ax.errorbar([low,high],p,yerr=[p-lo,hi-p],color='black',marker='o',capsize=4,lw=1)
        for x,y,top in zip([low,high],p,hi):ax.text(x,top+3,f'{y:.2f}%',ha='center',fontsize=9)
        ax.set(xticks=[low,high],xlim=(low-.45,high+.45),ylim=(10,85),xlabel=predictor,title=question+': '+title)
    axes[0].set_ylabel('Adjusted probability (%)');fig.tight_layout();save_figure(root,'02_main_probability_contrasts',fig);plt.close(fig)

    distributions=read(root,'02_measurement','outcome_distributions')
    fig,ax=plt.subplots(figsize=(7,4));x=np.arange(3)
    for i,outcome in enumerate(['N1','N3']):
        t=distributions.loc[distributions.outcome.eq(outcome)].set_index('category').loc[['Decreased','Unchanged','Increased']]
        bars=ax.bar(x+(i-.5)*.34,t.percent,width=.34,color='0.65' if i==0 else 'white',edgecolor='black',hatch='' if i==0 else '///',
                    label='Academic performance (N1)' if i==0 else 'Unaided self-study (N3)')
        ax.bar_label(bars,fmt='%.1f',padding=3,fontsize=9)
    ax.set(xticks=x,xticklabels=['Decreased','Unchanged','Increased'],ylabel='Students (%)',ylim=(0,78));ax.legend(frameon=False)
    fig.tight_layout();save_figure(root,'03_outcome_distributions',fig);plt.close(fig)

    pairs=read(root,'02_measurement','cau_aor_score_pairs');rho=read(root,'04_uncertainty','cau_aor_correlation').iloc[0]
    fig,ax=plt.subplots(figsize=(6.7,4.5));ax.scatter(pairs.CAU,pairs.AOR,s=8+12*pairs.n,facecolors='none',edgecolors='black',lw=.7)
    ax.set(xlabel='Constructive AI use (CAU)',ylabel='AI overreliance (AOR)',xlim=(.8,5.2),ylim=(.8,5.2),
           title=f'Spearman rho = {rho.rho:.3f}; n = 323')
    fig.tight_layout();save_figure(root,'04_cau_aor_score_pairs',fig);plt.close(fig)

    c3=read(root,'03_associations','c3_sensitivity').merge(boot.loc[boot.group.eq('C3')],on=['group','scenario','outcome'],validate='one_to_one')
    fig,axes=plt.subplots(1,2,figsize=(10,5),sharey=True);y=np.arange(len(c3))
    good=c3.OR_per_SD.notna();t=c3.loc[good];yy=y[good]
    axes[0].errorbar(t.OR_per_SD,yy,xerr=[t.OR_per_SD-t.OR_CI95_low,t.OR_CI95_high-t.OR_per_SD],fmt='o',color='black',capsize=3)
    axes[0].axvline(1,color='0.5',ls='--');axes[0].set_xlabel('OR per SD; 95% profile CI')
    axes[1].errorbar(c3.difference_pp,y,xerr=[c3.difference_pp-c3.difference_pp_CI95_low,c3.difference_pp_CI95_high-c3.difference_pp],fmt='o',color='black',capsize=3)
    axes[1].axvline(0,color='0.5',ls='--');axes[1].set_xlabel('Difference (pp); 95% bootstrap CI')
    axes[0].set_yticks(y,[LABELS[s] for s in c3.scenario]);axes[0].invert_yaxis();fig.tight_layout()
    save_figure(root,'05_c3_sensitivity',fig);plt.close(fig)

    bins=read(root,'05_prediction','calibration_bins');fig,axes=plt.subplots(1,2,figsize=(8.5,4),sharey=True)
    for ax,outcome,title in zip(axes,['N1','N3'],['Academic improvement','Unaided self-study improvement']):
        ax.plot([0,1],[0,1],color='0.6',ls='--')
        for model,style in [('CQ','o-'),('CS','s:')]:
            t=bins.loc[bins.outcome.eq(outcome)&bins.model.eq(model)].sort_values('mean_predicted')
            ax.plot(t.mean_predicted,t.observed_rate,style,color='black',label=model)
        ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean OOF predicted probability',title=title);ax.legend(frameon=False)
    axes[0].set_ylabel('Observed improvement proportion');fig.tight_layout();save_figure(root,'06_oof_calibration',fig);plt.close(fig)

    context=read(root,'06_interaction_context','school_classroom_context').set_index('item')
    ids=['AIC3','AIC2','RAI3','AIC5','TAG5','TAG3'];t=context.loc[ids]
    labels=['Responsible-use encouragement','Consistent teacher views','Peers disclose AI use','School AI activities','Dialogue with teachers','Guidance supports thinking']
    fig,ax=plt.subplots(figsize=(8.5,4));left=np.zeros(len(t))
    for field,color,hatch,label in [('agree_percent','0.3','','Agree'),('neutral_percent','0.7','','Neutral / uncertain'),('disagree_percent','white','///','Disagree')]:
        ax.barh(labels,t[field],left=left,color=color,edgecolor='black',hatch=hatch,label=label);left+=t[field].to_numpy()
    ax.invert_yaxis();ax.set(xlim=(0,100),xlabel='Students (%)');ax.legend(frameon=False,ncol=3,loc='lower center',bbox_to_anchor=(.5,1.02))
    fig.tight_layout();save_figure(root,'07_school_context',fig);plt.close(fig)
    finish_phase(root,PHASE)

    index=['# Recomputed tables and figures','',
           'All numerical tables below are generated from the input CSV. See the root README for interpretation and provenance.','',
           '[Download all tables](downloads/all_tables.zip)','']
    html_parts=['<!doctype html><html lang="en"><meta charset="utf-8"><title>Recomputed report tables</title>',
                '<style>body{font:15px Georgia,serif;max-width:1250px;margin:32px auto;color:#111}table{border-collapse:collapse;width:100%;margin:18px 0 40px}th,td{border-bottom:1px solid #bbb;padding:6px;text-align:left}th{background:#eee}section{overflow:auto}a{color:#111}</style><h1>Recomputed report tables</h1>']
    for folder in sorted((root/'tables').iterdir()):
        if not folder.is_dir():continue
        index.extend(['## '+folder.name,'',f'[Download phase tables](downloads/{folder.name}_tables.zip)',''])
        for p in sorted(folder.glob('*.csv')):
            index.append(f'- [{p.stem}](tables/{folder.name}/{p.name})')
            t=pd.read_csv(p);html_parts.append('<section><h2>'+html.escape(folder.name+' / '+p.stem)+'</h2>'+t.to_html(index=False,float_format=lambda x:f'{x:.6g}',escape=True)+'</section>')
        index.append('')
    (root/'index.md').write_text('\n'.join(index)+'\n')
    (root/'tables.html').write_text('\n'.join(html_parts)+'\n</html>')
    with zipfile.ZipFile(root/'downloads/all_tables.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for p in sorted((root/'tables').rglob('*.csv')):archive.write(p,str(p.relative_to(root)))
    return main
