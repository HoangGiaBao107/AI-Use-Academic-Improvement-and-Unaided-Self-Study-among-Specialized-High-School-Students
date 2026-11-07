"""Portable paths, table exports, and phase-level downloads."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,zipfile
import pandas as pd

def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=True,default=str)+'\n')

def save_table(root,phase,name,table):
    folder=Path(root)/'tables'/phase;folder.mkdir(parents=True,exist_ok=True)
    path=folder/(name+'.csv');table.to_csv(path,index=False,float_format='%.12g')
    return path

def finish_phase(root,phase):
    root=Path(root);folder=root/'downloads';folder.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(folder/(phase+'_tables.zip'),'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted((root/'tables'/phase).glob('*.csv')):
            z.write(p,'tables/'+p.name)

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save_figure(root,name,figure):
    folder=Path(root)/'figures';folder.mkdir(parents=True,exist_ok=True)
    figure.savefig(folder/(name+'.svg'),bbox_inches='tight',metadata={'Date':None})
    figure.savefig(folder/(name+'.png'),dpi=170,bbox_inches='tight')

def run_manifest(root,data_hash,args):
    import platform,numpy,pandas,scipy,sklearn,matplotlib,statsmodels,patsy
    write_json(Path(root)/'run_manifest.json',dict(
        input_sha256=data_hash,run_utc=datetime.now(timezone.utc).isoformat(),
        bootstrap=args.bootstrap,null_draws=args.null_draws,
        analysis='Exploratory same-cohort associations; not causal or externally validated.',
        production_settings=bool(args.bootstrap==2000 and args.null_draws==999),
        software={m.__name__:m.__version__ for m in [numpy,pandas,scipy,sklearn,matplotlib,statsmodels,patsy]},
        python=platform.python_version()))
