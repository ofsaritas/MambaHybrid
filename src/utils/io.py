from pathlib import Path
import json

def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)
def write_json(obj,p):
    ensure_dir(Path(p).parent)
    with open(p,'w',encoding='utf-8') as f: json.dump(obj,f,indent=2,ensure_ascii=False)
def read_json(p):
    with open(p,'r',encoding='utf-8') as f: return json.load(f)
