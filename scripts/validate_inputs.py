#!/usr/bin/env python3
from pathlib import Path
import argparse, json, yaml
from doubleadapt.data import load_panel
p=argparse.ArgumentParser(); p.add_argument("--input",required=True); p.add_argument("--config",required=True); a=p.parse_args()
frame=load_panel(a.input,True)
print(json.dumps({"status":"valid","rows":len(frame),"dates":int(frame.date.nunique()),"features":len([c for c in frame.columns if c.startswith("feature_")])},ensure_ascii=False))
