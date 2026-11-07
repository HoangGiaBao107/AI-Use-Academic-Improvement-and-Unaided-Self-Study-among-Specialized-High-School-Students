"""Refresh the published directory map after adding aggregate artifacts."""
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
EXCLUDED={'.git','.work','__pycache__','.venv'}

def tree(folder,depth=0):
    lines=[]
    for p in sorted(folder.iterdir(),key=lambda p:(not p.is_dir(),p.name.lower())):
        if p.name in EXCLUDED:continue
        lines.append('  '*depth+p.name+('/' if p.is_dir() else ''))
        if p.is_dir():lines.extend(tree(p,depth+1))
    return lines

if __name__=='__main__':
    target=ROOT/'docs/REPOSITORY_TREE.md'
    target.write_text('# Repository structure\n\nTracked research code, provenance, and aggregate outputs. The manuscript is added separately by the author.\n\n```text\n'+
                      'ai-learning-study/\n'+'\n'.join(tree(ROOT,1))+'\n```\n\nLocal-only input: `data/data raw eng.csv`. Private intermediates and validation replay files live in `.work/`.\n')
