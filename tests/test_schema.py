import pandas as pd
from doubleadapt.data import validate_panel
def test_schema():
    frame=pd.DataFrame({"date":[1,1],"ticker":["a","b"],"feature_1":[0.1,0.2],"label":[0.0,1.0]})
    assert validate_panel(frame)==["feature_1"]
