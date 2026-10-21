"""Create and validate the complete version 1.0 technical-review baseline."""
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import json

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'results/freeze-v1'
with ZipFile(ROOT/'output/evidence/MTR_KV_Revised_Evidence.zip') as previous:
    files={name:previous.read(name) for name in previous.namelist()
           if name!='manifest.json' and not name.startswith('disclosure/')}
    for path in (ROOT/'mtr_kv').glob('*.py'):
        assert path.read_bytes()==previous.read('code/mtr_kv/'+path.name),path

for folder in ['mtr_kv','tests','configs','frontend','experiments','scripts']:
    for path in (ROOT/folder).rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix in ['.py','.json','.js','.html','.css','.md']:
            files['code/'+str(path.relative_to(ROOT))]=path.read_bytes()
files['code/README.md']=(ROOT/'README.md').read_bytes()
for path in (ROOT/'docs').rglob('*.md'):files['notes/'+str(path.relative_to(ROOT/'docs'))]=path.read_bytes()
for path in DATA.iterdir():
    if path.is_file() and path.name!='freeze-deliverables.json':
        files['measurements/freeze-v1/'+path.name]=path.read_bytes()
files['input/big-city-life-original.webm']=(ROOT/'results/big-city-life-original.webm').read_bytes()
# The derivative is already included under measurements/freeze-v1 with its provenance.
for path in (ROOT/'output/evidence').glob('*.png'):files['screenshots/'+path.name]=path.read_bytes()
docx=ROOT/'output/docx/MTR_KV_Patent_Disclosure_v1_0.docx'
pdf=ROOT/'output/pdf/MTR_KV_Patent_Disclosure_v1_0.pdf'
for path in [docx,pdf]:files['disclosure/'+path.name]=path.read_bytes()
files['README.txt']=b'''MTR-KV version 1.0 technical disclosure and evidence baseline, 21 October 2026.
Run python3 verify_evidence.py after extracting to verify every recorded file hash.
The disclosure/ folder contains the matching DOCX and PDF. Preserve these bytes;
make paraphrases and later changes as a new version rather than replacing 1.0.
This is a review baseline, not a patent-office filing or a trusted timestamp.

measurements/ contains the original controlled 216-frame trace, six paired runs,
and the 128-pass test report. The runtime module bytes match the tested build.
measurements/freeze-v1/ contains the three new screenshot-session traces, their
capture-manifest.json, natural-video provenance, both natural traces/evaluations,
and the normalized H264 input. Screenshot timings are not primary benchmarks.
Capture manifest paths refer to the working project: output/evidence corresponds
to screenshots/ here; results/freeze-v1 corresponds to measurements/freeze-v1/.
The three captured endpoint decisions/reuse records match the primary serial run.

The natural input is Big City Life by Coverr, 2015, Wikimedia Commons, listed CC0:
https://commons.wikimedia.org/wiki/File:Big_City_Life.webm
Original VP8 input is under input/. Its SHA1 matches the Commons checksum.
The H264 derivative is 640x360 at 12fps, CRF20, medium preset, GOP48, no B frames,
with audio omitted. Sampling times and both SHA256 values are in the provenance.
Original: 356/356 READY; 100% metadata fallback; zero reuse; 356/356 top1 agreement.
Derivative: 171/171 READY; zero reuse; 170/171 top1 agreement. Frame124 differs.
These single natural-scene runs have no labels and do not prove general accuracy,
reusability, real-time throughput, or optimization speedup across devices.

For measurement reproduction create an environment in code/, install
requirements-live.txt and pytest, then run the archived tests with python -m pytest.
Model weights may need downloading. Use python -m mtr_kv video <input-path>
--semantic-model --kv-inference --kv-reuse --compare-baseline to write a new trace.
For the paired study from code/: PYTHONPATH=. python experiments/benchmark_optimization.py
--input ../input/generated-demo.mp4 --output ../new-replication-directory
For natural-video conversion, copy input/big-city-life-original.webm into code/results/,
then run experiments/validate_real_video.py --prepare-only followed by the same
script without the prepare flag. Document authoring also needs the retained Aura
reference and local authoring assets; the reference document is not redistributed.
No physical phone, GPU/RTL controller or full VLM integration is validated here.
'''
provenance=json.loads(files['measurements/freeze-v1/natural-video-provenance.json'])
assert sha256(files['input/big-city-life-original.webm']).hexdigest()==provenance['original_sha256']
assert sha256(files['measurements/freeze-v1/big-city-life-640x360-12fps-h264.mp4']).hexdigest()==provenance['derivative_sha256']
captures=json.loads(files['measurements/freeze-v1/capture-manifest.json'])
for item in captures['captures']:
    assert sha256(files['screenshots/'+Path(item['screenshot']).name]).hexdigest()==item['screenshot_sha256']
    raw=files['measurements/freeze-v1/'+Path(item['trace']).name]
    assert sha256(raw).hexdigest()==item['trace_sha256']
    assert json.loads(raw.splitlines()[-1])==item['record']
manifest={'kind':'MTR_KV_frozen_technical_review_baseline','version':'1.0',
          'prepared_local_date':'2026-10-21',
          'freeze_meaning':'Versioned byte-identifiable review baseline; later edits require a new version.',
          'files':{n:sha256(data).hexdigest() for n,data in sorted(files.items())}}
files['manifest.json']=json.dumps(manifest,indent=2).encode()
archive=ROOT/'output/evidence/MTR_KV_Frozen_Evidence_v1_0.zip'
with ZipFile(archive,'w',ZIP_DEFLATED) as out:
    for name,data in sorted(files.items()):out.writestr(name,data)
with ZipFile(archive) as out:
    for name,digest in manifest['files'].items():assert sha256(out.read(name)).hexdigest()==digest,name
hashes={str(path.relative_to(ROOT)):sha256(path.read_bytes()).hexdigest() for path in [docx,pdf,archive]}
(ROOT/'output/evidence/MTR_KV_v1_0_SHA256SUMS.txt').write_text(''.join(digest+'  '+name+'\n' for name,digest in hashes.items()))
(DATA/'freeze-deliverables.json').write_text(json.dumps({'version':'1.0','files':hashes,'archive_verified_files':len(manifest['files'])},indent=2))
print(archive,'verified',len(manifest['files']),'files; runtime source unchanged; screenshot and input provenance verified')
