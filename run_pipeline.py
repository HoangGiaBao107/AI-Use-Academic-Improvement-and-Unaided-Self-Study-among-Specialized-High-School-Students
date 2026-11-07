"""Run report-linked phases from the reviewed English CSV."""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[name]='1'
from pathlib import Path
import argparse,sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ROOT/'data/data raw eng.csv')
    p.add_argument('--output',type=Path,default=ROOT/'results')
    p.add_argument('--phases',default='1,2,3,4,5,6,7',help='Comma-separated phase numbers.')
    p.add_argument('--bootstrap',type=int,default=2000)
    p.add_argument('--null-draws',type=int,default=999)
    p.add_argument('--workers',type=int,default=4)
    return p

def main(argv=None):
    args=parser().parse_args(argv);args.repo=ROOT
    args.output=args.output.resolve();args.data=args.data.resolve()
    phases=[int(n) for n in args.phases.split(',')]
    if min(args.bootstrap,args.null_draws,args.workers)<1:
        raise ValueError('Bootstrap draws, null draws, and workers must be positive.')
    if any(n not in range(1,8) for n in phases):raise ValueError('Phase numbers must be between 1 and 7.')
    if not args.data.is_file():
        other=ROOT/'data raw eng.csv'
        if other.is_file():args.data=other
        else:raise FileNotFoundError('Place data raw eng.csv in data/ or supply --data PATH. See data/README.md.')
    from ai_learning.data import load_analysis
    from ai_learning.storage import run_manifest
    _,_,_,_,sha=load_analysis(args.data)
    for phase in phases:
        print(f'PHASE {phase}',flush=True)
        if phase==1:
            from ai_learning.measurement import prepare as run
        elif phase==2:
            from ai_learning.measurement import describe as run
        elif phase==3:
            from ai_learning.associations import run
        elif phase==4:
            from ai_learning.uncertainty import run
        elif phase==5:
            from ai_learning.prediction import run
        elif phase==6:
            from ai_learning.interaction import run
        else:
            from ai_learning.reporting import run
        run(args)
    run_manifest(args.output,sha,args)
    print('Completed:',args.output,flush=True)

if __name__=='__main__':main()
