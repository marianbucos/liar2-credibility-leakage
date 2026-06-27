"""
Downloads the original LIAR dataset from the UCSB source and saves it as
train.tsv / test.tsv / valid.tsv in ../liar/ (relative to this script).
"""
import io
import urllib.request
import zipfile
from pathlib import Path

URL = 'https://www.cs.ucsb.edu/~william/data/liar_dataset.zip'
OUT_DIR = Path(__file__).parent.parent / 'data' / 'liar'
OUT_DIR.mkdir(exist_ok=True)

FILES = ['train.tsv', 'test.tsv', 'valid.tsv']

print(f'Downloading {URL} ...')
with urllib.request.urlopen(URL) as resp:
    data = resp.read()

print('Extracting ...')
with zipfile.ZipFile(io.BytesIO(data)) as zf:
    for filename in FILES:
        content = zf.read(filename)
        out_path = OUT_DIR / filename
        out_path.write_bytes(content)
        rows = content.count(b'\n')
        print(f'Saved {rows} rows -> {out_path}')

print('Done.')
