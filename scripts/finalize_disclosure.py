"""Fill the original IDF-B pages, append the verified annex, package evidence."""
from hashlib import sha256
import io
import json
from pathlib import Path
import shutil
import zipfile

from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.colors import HexColor

ROOT=Path(__file__).resolve().parents[1]
TEMPLATE=Path('/Users/shaurya/Downloads/Invention_Disclosure_Format_B.pdf')
ANNEX=ROOT/'tmp/disclosure/render/MTR_KV_IDF_B_Detailed.pdf'
FINAL=ROOT/'output/pdf/MTR_KV_IDF_B_Completed.pdf'
DATA=ROOT/'results/optimization-20261021'


def fill_original():
    stream=io.BytesIO()
    drawing=canvas.Canvas(stream,pagesize=(595.32,841.92))
    style=ParagraphStyle('answer',fontName='Helvetica',fontSize=9.5,leading=11.7,textColor=HexColor('#163548'))
    answers=[
        (178,35,'MTR-KV: generation-safe motion-guided residency control for visual key-value memory. Detailed title and field: Annex A1.'),
        (230,40,'Computer architecture and systems software for video-transformer inference, compressed-domain metadata, visual-token/page mapping, tiered KV memory, semantic correction and bounded temporal reuse. See Annex A1.'),
        (296,60,'Reviewed: PagedAttention, CodecSight, CoPE-VideoLM, RESUME, H2O and SnapKV; decoder-memory patents and transformer KV patents R7-R11. The detailed comparison table, overlaps and candidate distinctions are in Annex A2-A5; complete references are in A20-A21. This is a focused search, not an exhaustive novelty opinion.'),
        (388,48,'Codec evidence proposes page-tier intent before attention. A generation-checked reserve/prepare/commit/ready protocol gates use; later model attention corrects placement through the same controller. Exact final-block feature identity additionally qualifies bounded reuse. Candidate contribution and prior-art boundaries: Annex A4-A5.'),
        (460,46,'Reduce unnecessary fast-tier residency; reconcile codec and token geometry; prevent stale mapping/correction updates; rescue important quiet pages; safely retain unchanged K/V; and expose reproducible measurements. Objectives and working principle: Annex A6.'),
        (537,48,'Decode and normalize motion; reconcile actual crop/patch/page geometry; update temporal state; emit MKAD; reserve, prepare, commit and signal ready; attach or validate/reuse K/V; run attention; apply current-generation semantic correction. Missing metadata causes conservative fallback. See Annex A6-A12.'),
        (613,49,'See Annex A7-A12 for the full system, geometry, descriptor, transaction, tensor-store and reuse descriptions, including Figures 1-4. The executable embodiment uses a real H.264 decoder and the final ViT-Tiny attention block. GPU/RTL KMC and full VLM integration remain proposed.'),
        (694,48,'128 tests passed. Paired median frame time: 55.57 ms oracle vs 26.48 ms indexed (3 pairs, matched decisions). Full generated-video run: 216 ready frames, 862 page reuses, 100% top-1 agreement. CPU/sample-specific evidence, not VLM or production speedup proof. Results, screenshots and hashes: Annex A13-A17.'),
        (767,48,'Protection is sought for the specific codec-to-page, generation-safe readiness and same-controller semantic correction arrangement, with explicitly qualified temporal payload reuse. Generic paging, motion vectors, quantization, attention scores and reuse are not asserted as new alone. Technical topics for agent review: Annex A18.'),
    ]
    for top,height,text in answers:
        item=Paragraph(text,style)
        width,actual=item.wrap(486,height)
        if actual>height:
            raise ValueError(f'Form answer overflow at top {top}: {actual} > {height}')
        item.drawOn(drawing,55,841.92-top-actual)
    drawing.showPage()
    # Mark the supplied TRL 3 checkbox row, preserving the original table.
    drawing.setStrokeColor(HexColor('#163548'));drawing.setLineWidth(2)
    drawing.line(180,841.92-389,193,841.92-402)
    drawing.line(193,841.92-389,180,841.92-402)
    notes=[
        ('Selected TRL 3 - experimental proof of concept',True),
        ('The complete disclosed architecture is conservatively self-assessed at TRL 3. Software modules have local lab validation, including real encoded video and actual tensor consumption; physical GPU/RTL and full VLM integration have not been demonstrated. Detailed assessment: Annex A19.',False),
        ('Detailed annex follows as an integral part of this completed disclosure',True),
        ('The original form and its printed ending are preserved. Annex A1-A22 provides all expanded answers, prior-work tables, five technical/result figures, two authentic prototype screenshots, measured outcomes, evidence locations and references.',False),
        ('Disclosure facts and confirmation',True),
        ('Prepared 21 October 2026. The user reported no known public posting, presentation, submission or patent filing. Legal inventorship, ownership and institutional requirements still require confirmation. No signature or legal declaration is supplied by this document.',False),
    ]
    top=474
    for text,bold in notes:
        item=Paragraph(('<b>'+text+'</b>') if bold else text,style)
        _,height=item.wrap(486,200)
        item.drawOn(drawing,55,841.92-top-height)
        top+=height+10
    drawing.save();stream.seek(0)
    overlay=PdfReader(stream)
    original=PdfReader(str(TEMPLATE))
    writer=PdfWriter()
    for index,page in enumerate(original.pages):
        page.merge_page(overlay.pages[index])
        writer.add_page(page)
    annex=PdfReader(str(ANNEX))
    if len(annex.pages)!=22:
        raise ValueError('Annex pagination changed; update form references before shipping')
    for page in annex.pages:writer.add_page(page)
    writer.add_metadata({'/Title':'MTR-KV Completed Invention Disclosure Format B',
                         '/Subject':'Technical disclosure with measured software prototype evidence',
                         '/Author':'Prepared for inventor and VIT IPR review'})
    writer.add_outline_item('Completed original IDF-B form',0)
    for index,title in enumerate(json.loads((ROOT/'tmp/disclosure/page-titles.json').read_text())):
        writer.add_outline_item(f'Annex A{index+1} {title}',index+2)
    FINAL.parent.mkdir(parents=True,exist_ok=True)
    with FINAL.open('wb') as out:writer.write(out)
    reopened=PdfReader(str(FINAL))
    assert len(reopened.pages)==24
    all_text='\n'.join(page.extract_text() or '' for page in reopened.pages)
    for required in ['128','862','55.57','26.48','TRL 3','CodecSight','US20260080217A1']:
        assert required in all_text,required
    print(FINAL)


def package():
    files={}
    def add(path,name):files[name]=Path(path).read_bytes()
    add(TEMPLATE,'source/Invention_Disclosure_Format_B.pdf')
    add(ROOT/'docs/MTR_KV_Frozen_Master_Architecture_and_Prototype_Specification_FINAL_v1_0.docx','source/Frozen_Architecture_v1.docx')
    add(ROOT/'results/demo.mp4','input/generated-demo.mp4')
    for name in ['full-demo-serial.jsonl','full-demo-serial-evaluation.json','test-results.txt','test-results.xml']:
        add(DATA/name,'measurements/'+name)
    for path in sorted((DATA/'paired-serial').glob('*')):
        if path.is_file():add(path,'measurements/paired-serial/'+path.name)
    for path in sorted((ROOT/'output/evidence').glob('*.png')):add(path,'screenshots/'+path.name)
    add(ROOT/'output/evidence/runtime-and-provenance.json','runtime-and-provenance.json')
    for directory in ['mtr_kv','tests','configs','frontend','experiments','scripts']:
        for path in sorted((ROOT/directory).rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix in ('.py','.json','.html','.css','.js','.md'):
                add(path,'code/'+str(path.relative_to(ROOT)))
    add(ROOT/'requirements-live.txt','code/requirements-live.txt')
    add(ROOT/'README.md','code/README.md')
    for path in sorted((ROOT/'tmp/disclosure/assets').glob('*.svg')):add(path,'figures/'+path.name)
    for path in [ROOT/'docs/KV_REUSE.md',ROOT/'docs/PHONE_CAMERA.md',ROOT/'docs/OPTIMIZATION.md']:
        if path.exists():add(path,'notes/'+path.name)
    references=[
        ('R1','https://arxiv.org/html/2309.06180v1'),('R2','https://arxiv.org/html/2604.06036v4'),
        ('R3','https://arxiv.org/html/2602.13191v2'),('R4','https://arxiv.org/html/2609.39563v1'),
        ('R5','https://arxiv.org/abs/2306.14048'),('R6','https://arxiv.org/abs/2404.14469'),
        ('R7','https://patents.google.com/patent/US6072548A/en'),('R8','https://patents.google.com/patent/US8300697B2/en'),
        ('R9','https://patents.google.com/patent/US20250124105A1/en'),('R10','https://patents.google.com/patent/US20250225134A1/en'),
        ('R11','https://patents.google.com/patent/US20260080217A1/en'),
        ('R12','https://ipindia.gov.in/acts/patent-act-1970/section-10'),('R13','https://www.wipo.int/en/web/patents/faq_patents')]
    files['references-and-search.json']=json.dumps({'accessed_local_date':'2026-10-21','scope':'Focused primary-source review, not exhaustive patentability/FTO search',
        'sources':[{'reference':key,'url':url} for key,url in references],
        'queries':['motion vector transformer KV cache page table','codec metadata descriptor memory controller attention',
                   'video decoder semantic cache correction generation','visual token page residency motion residual',
                   'KV cache generation motion controller','attention guided KV retention'],
        'remaining_review':'Patent family priority, full claim interpretation, additional jurisdictions, citations and combinations; no negative novelty guarantee.'},indent=2).encode()
    verify='''from hashlib import sha256
import json
from pathlib import Path
root=Path(__file__).resolve().parent
manifest=json.loads((root/'manifest.json').read_text())
for path,expected in manifest['files'].items():
    assert sha256((root/path).read_bytes()).hexdigest()==expected, path
print('Verified',len(manifest['files']),'files')
'''
    files['verify_evidence.py']=verify.encode()
    files['README.txt']=b'''MTR-KV disclosure evidence prepared 21 October 2026.
Extract this archive and run python3 verify_evidence.py to verify file hashes.
Generated input is a controlled demonstration, not a representative dataset.
The paired benchmark uses final-code oracle/indexed paths with identical crop,
storage, model and reuse settings. First 6 of 60 frames are excluded per run.
The full serial trace covers 216 frames. Test report: 128 passes.
Screenshots are actual local browser captures; they are not benchmark runs.
Hashes identify byte content, not independent authenticity or patent priority.
No physical phone, GPU controller, full VLM or production deployment is proved.
Code: create a Python environment, install requirements-live.txt and pytest, then run pytest.
Replicate from code/: PYTHONPATH=. python experiments/benchmark_optimization.py
--input ../input/generated-demo.mp4 --output ../replication-new-directory
Software/runtime packages are not embedded in this archive.
'''
    manifest={'kind':'MTR_KV_IDF_B_evidence','prepared_local_date':'2026-10-21',
              'files':{name:sha256(data).hexdigest() for name,data in sorted(files.items())}}
    files['manifest.json']=json.dumps(manifest,indent=2).encode()
    archive=ROOT/'output/evidence/MTR_KV_Evidence.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name,data in sorted(files.items()):z.writestr(name,data)
    with zipfile.ZipFile(archive) as z:
        for name,digest in manifest['files'].items():assert sha256(z.read(name)).hexdigest()==digest
    (ROOT/'output/evidence/deliverable-hashes.json').write_text(json.dumps({
        str(path.relative_to(ROOT)):sha256(path.read_bytes()).hexdigest()
        for path in [FINAL,ROOT/'output/docx/MTR_KV_IDF_B_Detailed.docx',archive]},indent=2))
    print(archive,'verified files',len(manifest['files']))


if __name__=='__main__':
    fill_original()
    package()
