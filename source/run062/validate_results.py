from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
import validate as frozen

ORIGINAL=json.dumps
def safe(v:Any)->Any:
    if isinstance(v,dict): return {str(k):safe(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)): return [safe(x) for x in v]
    if isinstance(v,np.generic): return v.item()
    return v
def dumps(v:Any,*a:Any,**kw:Any)->str: return ORIGINAL(safe(v),*a,**kw)
def main()->None:
    frozen.json.dumps=dumps
    frozen.main()
if __name__=='__main__': main()
