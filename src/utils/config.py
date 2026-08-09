import yaml
from pathlib import Path

def load_config(path):
    with open(path,'r',encoding='utf-8') as f:
        return yaml.safe_load(f)

def save_config(cfg, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path,'w',encoding='utf-8') as f:
        yaml.safe_dump(cfg,f,sort_keys=False)
