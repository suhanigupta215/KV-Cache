"""Prepare and evaluate a provenance-recorded natural-scene video, without quality claims."""
from fractions import Fraction
from hashlib import sha1, sha256
from pathlib import Path
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import av
from mtr_kv.config import load_config
from mtr_kv.video import run_video
from mtr_kv.evaluate import evaluate_file

OUT=ROOT/'results/freeze-v1'
OUT.mkdir(exist_ok=True)
original=ROOT/'results/big-city-life-original.webm'
derived=OUT/'big-city-life-640x360-12fps-h264.mp4'

def prepare():
    assert sha1(original.read_bytes()).hexdigest()=='767e093934208a8d9124d22fd3f517d87be65c81'
    sample_times=[]
    with av.open(str(original)) as source, av.open(str(derived),'w') as target:
        stream=target.add_stream('libx264',rate=12)
        stream.width=640;stream.height=360;stream.pix_fmt='yuv420p'
        stream.options={'crf':'20','preset':'medium','g':'48','bf':'0'}
        info={'codec':source.streams.video[0].codec_context.name,
              'width':source.streams.video[0].width,'height':source.streams.video[0].height,
              'rate':str(source.streams.video[0].average_rate)}
        for frame in source.decode(video=0):
            timestamp=float(frame.time)
            if timestamp+1e-6<len(sample_times)/12:continue
            sample_times.append(timestamp)
            resized=frame.reformat(width=640,height=360,format='yuv420p')
            resized.pts=len(sample_times)-1;resized.time_base=Fraction(1,12)
            for packet in stream.encode(resized):target.mux(packet)
        for packet in stream.encode():target.mux(packet)
    provenance={'title':'Big City Life','author_credit':'Coverr',
        'source_page':'https://commons.wikimedia.org/wiki/File:Big_City_Life.webm',
        'media_url':'https://upload.wikimedia.org/wikipedia/commons/2/2a/Big_City_Life.webm',
        'license_as_listed':'CC0 1.0 Universal Public Domain Dedication',
        'license_url':'https://creativecommons.org/publicdomain/zero/1.0/',
        'source_page_revision':'823488287','retrieved_local_date':'2026-10-21',
        'original_sha1':sha1(original.read_bytes()).hexdigest(),
        'original_sha256':sha256(original.read_bytes()).hexdigest(),'original_video':info,
        'derivative_sha256':sha256(derived.read_bytes()).hexdigest(),
        'derivative':{'width':640,'height':360,'fps':12,'codec':'h264','encoder':'libx264',
                      'crf':20,'preset':'medium','gop':48,'b_frames':0,'audio':'omitted',
                      'frames':len(sample_times),'sampled_original_seconds':sample_times},
        'scope':'One natural-scene convenience sample; no task labels or representativeness claim. H264 motion metadata depends on this re-encoding.'}
    (OUT/'natural-video-provenance.json').write_text(json.dumps(provenance,indent=2))
    print('Prepared',len(sample_times),'frames',flush=True)

def evaluate():
    for label,path in [('original-vp8',original),('derived-h264',derived)]:
        trace=OUT/(label+'.jsonl')
        with trace.open('w') as sink:
            run_video(load_config(str(ROOT/'configs/reference.json')),str(path),sink,
                semantic_model=True,kv_inference=True,kv_reuse=True,compare_baseline=True)
        result=evaluate_file(str(trace),str(OUT/(label+'-evaluation.json')))
        print(result,flush=True)

if __name__=='__main__':
    if '--prepare-only' in sys.argv:prepare()
    else:evaluate()
