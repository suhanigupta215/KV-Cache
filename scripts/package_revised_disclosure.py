"""Package the revised disclosure with the measured implementation evidence."""
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import json

ROOT=Path(__file__).resolve().parents[1]
with ZipFile(ROOT/'output/evidence/MTR_KV_Evidence.zip') as original:
    entries={n:original.read(n) for n in original.namelist() if n!='manifest.json'}
for src,name in [
    ('output/docx/MTR_KV_Patent_Disclosure_Revised.docx','disclosure/MTR_KV_Patent_Disclosure_Revised.docx'),
    ('output/pdf/MTR_KV_Patent_Disclosure_Revised.pdf','disclosure/MTR_KV_Patent_Disclosure_Revised.pdf'),
    ('output/evidence/MTR_KV_Implementation_Records.json','MTR_KV_Implementation_Records.json'),
    ('tmp/aura-reference/assets/geometry-measured.svg','figures/geometry.svg'),
    ('tmp/aura-reference/assets/architecture.svg','figures/architecture.svg'),
    ('scripts/rebuild_patent_reference.py','code/scripts/rebuild_patent_reference.py'),
    ('scripts/package_revised_disclosure.py','code/scripts/package_revised_disclosure.py'),
]:entries[name]=(ROOT/src).read_bytes()
entries['README.txt']+=b'''
Revised disclosure organization follows the supplied IDF-B reference.
The disclosure/ directory contains the final revised DOCX and matching PDF.
MTR_KV_Implementation_Records.json contains selected unmodified frame records
from measurements/full-demo-serial.jsonl and the complete trace hash.
Figures/geometry.svg now shows the measured model Resize248/CenterCrop224,
not the separate regression-test Resize256 transform. The runtime and trace
measurements are unchanged. The architecture label was shortened for legibility.
The reference DOCX itself is not redistributed. Its content is not project proof.
The document builder requires the original supplied reference at its local path.
'''
manifest={'kind':'MTR_KV_revised_disclosure_evidence','prepared_local_date':'2026-10-21',
          'files':{n:sha256(data).hexdigest() for n,data in sorted(entries.items())}}
entries['manifest.json']=json.dumps(manifest,indent=2).encode()
output=ROOT/'output/evidence/MTR_KV_Revised_Evidence.zip'
with ZipFile(output,'w',ZIP_DEFLATED) as archive:
    for n,data in sorted(entries.items()):archive.writestr(n,data)
with ZipFile(output) as archive:
    for n,digest in manifest['files'].items():assert sha256(archive.read(n)).hexdigest()==digest,n
    records=json.loads(archive.read('MTR_KV_Implementation_Records.json'))
    raw=archive.read('measurements/full-demo-serial.jsonl')
    assert sha256(raw).hexdigest()==records['sha256']
    frames={x['generation']:x for x in map(json.loads,raw.splitlines())}
    assert all(frames[x['generation']]==x for x in records['records'])
print(output,'verified',len(manifest['files']),'files and selected frame records')
