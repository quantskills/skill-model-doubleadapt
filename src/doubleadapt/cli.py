from __future__ import annotations
import argparse, json
from pathlib import Path
import yaml
from .data import load_panel
from .model import DoubleAdaptModel
from .trainer import train_model
from .inference import load_checkpoint, predict

def train_main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); p.add_argument("--input",required=True); p.add_argument("--output-dir",required=True)
    a=p.parse_args(argv); cfg=yaml.safe_load(Path(a.config).read_text()); frame=load_panel(a.input,True)
    features=[c for c in frame.columns if c.startswith("feature_")]; mc=cfg["model"]
    model=DoubleAdaptModel(len(features),mc["hidden_size"],mc["layers"],mc["dropout"],mc["heads"],mc["temperature"])
    score=train_model(model,cfg["split"]["train_dates"],cfg["split"]["validation_dates"],frame,{**mc,"input_size":len(features)},a.output_dir)
    print(json.dumps({"status":"complete","best_valid_pearson":score},ensure_ascii=False)); return 0

def infer_main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); p.add_argument("--checkpoint",required=True); p.add_argument("--input",required=True); p.add_argument("--output",required=True)
    a=p.parse_args(argv); model,ck=load_checkpoint(a.checkpoint); frame=load_panel(a.input,False)
    out=predict(model,frame,ck["feature_names"]); Path(a.output).parent.mkdir(parents=True,exist_ok=True); out.to_parquet(a.output,index=False)
    print(json.dumps({"status":"complete","rows":len(out)},ensure_ascii=False)); return 0
