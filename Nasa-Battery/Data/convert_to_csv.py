import os
import scipy.io
import numpy as np
import pandas as pd

def matlab_to_dict(mat_struct):
    """Recursively convert a MATLAB struct (loaded by scipy) into a Python dict."""
    result = {}
    # fields are accessed as array of dtype=object
    for name in mat_struct.dtype.names:
        value = mat_struct[name][0, 0]  # matlab structs are 1x1 arrays
        if hasattr(value, "dtype") and value.dtype.names:            # another struct
            result[name] = matlab_to_dict(value)
        elif isinstance(value, np.ndarray) and value.size == 1:      # scalar
            result[name] = value.item()
        else:
            result[name] = value
    return result

def flatten_matvars(mat_data):
    """Extract non-__ keys and convert structs to dicts."""
    out = {}
    for k, v in mat_data.items():
        if k.startswith("__"):
            continue
        if hasattr(v, "dtype") and v.dtype.names:   # struct array
            out[k] = matlab_to_dict(v)
        else:
            out[k] = v
    return out

src_dir = "Data"
dst_dir = "processed_csv"
os.makedirs(dst_dir, exist_ok=True)

for fname in os.listdir(src_dir):
    if not fname.lower().endswith(".mat"):
        continue
    mat = scipy.io.loadmat(os.path.join(src_dir, fname))
    vars = flatten_matvars(mat)

    # inspect the structure and decide how to convert to tabular form
    # here we assume there's one main variable that's a 2‑D array
    # adjust this logic to suit your actual data
    if len(vars) == 1:
        key = next(iter(vars))
        arr = vars[key]
        if isinstance(arr, np.ndarray) and arr.ndim == 2:
            df = pd.DataFrame(arr)
        else:
            # if it’s a dict or nested structure, you might need to
            # normalize it first, e.g. with pandas.json_normalize
            df = pd.json_normalize(arr)
    else:
        # combine multiple variables into a single DataFrame
        df = pd.json_normalize(vars)


    outname = os.path.splitext(fname)[0] + "_processed.csv"
    df.to_csv(os.path.join(dst_dir, outname), index=False)
    print(f"wrote {outname}")