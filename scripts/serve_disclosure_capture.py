"""Run unchanged dashboard paths to exact evidence endpoints and save matching records."""
from pathlib import Path
import json
import sys
from threading import Thread

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mtr_kv.dashboard as dashboard

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/freeze-v1'
OUT.mkdir(parents=True,exist_ok=True)
original=dashboard.run_video

def recorded_run(*args,**kwargs):
    endpoint=kwargs['max_frames']
    preview=kwargs.get('on_frame')
    with (OUT/f'capture-through-{endpoint}.jsonl').open('w') as out:
        def record(result,frame):
            if preview:preview(result,frame)
            clean={k:v for k,v in result.items() if k!='preview_jpeg'}
            out.write(json.dumps(clean)+'\n');out.flush()
        kwargs['on_frame']=record
        return original(*args,**kwargs)

dashboard.run_video=recorded_run
servers=[]
for port,frames in [(8771,1),(8772,55),(8773,163)]:
    server=dashboard.create_server(str(ROOT/'results/demo.mp4'),port=port,
        max_frames=frames,semantic_model=True,kv_inference=True,kv_reuse=True,
        compare_baseline=True,realtime=False,config_path=str(ROOT/'configs/reference.json'))
    servers.append(server)
    Thread(target=server.serve_forever,daemon=True).start()
    print(f'Capture endpoint http://127.0.0.1:{port} through frame {frames}',flush=True)
try:
    __import__('threading').Event().wait()
finally:
    for server in servers:server.shutdown();server.server_close()
