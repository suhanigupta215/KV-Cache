"""Add trace-linked operating states and a natural-scene validation to review version 1.0."""
from pathlib import Path
import json
import shutil
import rebuild_patent_reference as b

ROOT=b.ROOT
DATA=ROOT/'results/freeze-v1'
b.OUT=ROOT/'output/docx/MTR_KV_Patent_Disclosure_v1_0.docx'
b.doc.core_properties.title='MTR KV Invention Disclosure Version 1 0'
capture=json.loads((DATA/'capture-manifest.json').read_text())
natural={key:json.loads((DATA/(key+'-evaluation.json')).read_text()) for key in ['original-vp8','derived-h264']}
provenance=json.loads((DATA/'natural-video-provenance.json').read_text())
records=[json.loads(line) for line in (DATA/'derived-h264.jsonl').read_text().splitlines()]
bad=[x for x in records if not x['baseline_comparison']['same_top1']]
assert len(bad)==1 and bad[0]['generation']==124
rawheading=b.heading
rawpara=b.para
rawemit=b.emit

def para(text,*args,**kwargs):
    if str(text).startswith('MTR_KV_Evidence.zip contains'):
        text='MTR_KV_Frozen_Evidence_v1_0.zip contains the controlled and natural-scene source inputs, full serial and paired traces, evaluations, test text/XML, five authentic UI screenshots, matching capture records, source/configuration snapshots, references, conversion provenance and a SHA-256 manifest. The archive includes this version of the disclosure. A verification script checks hashes; they identify bytes, not a trusted timestamp or independent certification.'
    result=rawpara(text,*args,**kwargs)
    if text=='Project name: MTR-KV':
        rawpara('Version 1.0    Technical disclosure and evidence baseline    21 October 2026')
    return result

def capture_pages():
    rawheading('8.7 Operating states captured from the running prototype')
    b.picture('prototype-fallback-frame-1','Figure 8 Missing metadata invokes conservative residency',
        'The actual decoder supplies zero motion vectors at frame 1. The controller returns READY with seven T0 pages, while K/V reuse remains zero and all seven pages are computed. Missing metadata is handled as uncertainty; it is not interpreted as a static scene.',
        'capture-through-1.jsonl, final generation 1; prototype-fallback-frame-1.png; capture-manifest.json.')
    b.table(['Matching frame record','Observed value'],[
        ['status / metadata_fallback','READY / true'],
        ['motion_vectors / pages_reused / pages_recomputed','0 / 0 / 7'],
        ['mapping / baseline comparison','7 T0 pages, all ready / top-1 agrees'],
    ],[3.3,3.8])
    rawpara('These screenshots are unmodified browser captures from three independent runs ending at the indicated generation. The capture helper observes the normal dashboard path and stores the exact records sent to the browser without preview JPEGs. Their controller decisions, semantic actions, page states and reuse provenance match the corresponding primary serial records. Displayed timing is an observation from the capture session, not a replacement benchmark.')

    b.newpage()
    b.picture('prototype-moving-frame-55','Figure 9 Moving input rejects apparently reusable pages',
        'Frame 55 starts the moving-object phase. The interface shows zero reused pages, seven refreshed pages and five T0 pages after semantic correction. The matching record identifies why the candidates were rejected; low codec activity alone cannot authorize reuse.',
        'capture-through-55.jsonl, final generation 55; prototype-moving-frame-55.png; capture-manifest.json.')
    b.table(['Matching frame record','Observed value'],[
        ['source_seconds / motion_vectors','4.5 / 902'],
        ['model_input_changed','5 pages rejected after exact feature comparison'],
        ['codec_or_temporal_invalidation','2 pages rejected by earlier eligibility checks'],
        ['pages_reused / pages_recomputed / top-1','0 / 7 / agrees'],
    ],[3.3,3.8])
    rawpara('The five model-input rejections demonstrate the second validation stage in the implemented reuse contract. Unchanged local codec evidence can coexist with changed contextual features because earlier transformer attention is global. This example is evidence that the guard executed, not a count of detection accuracy on an external task.')

    b.newpage()
    b.picture('prototype-scene-cut-frame-163','Figure 10 Scene change discards retained payload eligibility',
        'The source background changes at frame 163, at 13.5 seconds. The decoder exposes no usable motion vectors for that frame. Conservative fallback places all seven pages in T0, with zero pages reused and seven recomputed. The displayed cumulative fallback count is four; that count includes earlier frames.',
        'capture-through-163.jsonl, final generation 163; prototype-scene-cut-frame-163.png; capture-manifest.json.')
    b.table(['Matching frame record','Observed value'],[
        ['status / metadata_fallback / motion_vectors','READY / true / 0'],
        ['reuse rejection reason','codec_or_temporal_invalidation for all 7 pages'],
        ['pages_reused / pages_recomputed','0 / 7'],
        ['actual tensor payload / baseline top-1','150528 bytes / agrees'],
    ],[3.3,3.8])
    rawpara('The phase boundary is defined by the controlled source generator. This observation shows safe processing of that cut through fallback; it does not establish a general scene-cut detector or a measured scene-change recall rate.')

    b.newpage();rawheading('8.8 Feature to code test and evidence map')
    rawpara('Each row identifies an implemented mechanism, an executable entry point, a regression case and an observation in the supplied evidence. Code modules are under mtr_kv and regression files under tests in the archived code directory. Function names are exact source identifiers. The 128-pass JUnit report is the common test record.')
    b.table(['Feature','Code entry point','Regression evidence','Run or figure evidence'],[
        ['Source crop and page geometry','vision_feedback.py\nsource_crop\nmtre.py\nreconcile_optimized','test_geometry.py\ntest_model_center_crop_maps_token_bounds_to_actual_source_window\ntest_mtre_optimized_matches_oracle','Figure 2\nsource_crop in all frame records; paired decision hashes match'],
        ['Generation checked readiness','controller.py\nKmc.apply\n_generation_check\n_allocate','test_memory_generation.py\ntest_stale_descriptor_rejected\ntest_ready_implies_committed\ntest_exhaustion_preserves_old_mapping_and_ready_state','ready_tokens and apply_statuses in each captured frame; reservation failure is covered by tests'],
        ['Same generation semantic correction','live.py\nLiveController.semantic\nsemantic.py\nSrce.evaluate','test_live.py\ntest_live_semantic_feedback_promotes_current_quiet_page','Generation 3\npages 0, 5, 6: T2 to T0\nSection 8.6'],
        ['Actual tiered tensor access','tensor_store.py\nTorchKVStore.put / get\nsync_tiers','test_tensor_store.py\ntest_tensor_store_roundtrip_real_payloads\ntest_tier_sync_does_not_count_spurious_reads','kv_consumed_pages=7; actual tensor_storage fields in live and serial traces'],
        ['Exact input reuse and age bound','vision_feedback.py\n_reuse_attention_kv\ntensor_store.py\nreuse','test_kv_reuse.py\ntest_changed_model_input_is_recomputed_despite_zero_codec_motion\ntest_window_refreshes_old_origins','Figure 9\n5 changed-input rejections; generation 43 window_expired; generation 8 reuse'],
        ['Metadata fallback','video.py\nrun_video\nlive.py\nLiveController.frame','test_reference_cases.py\ntest_T08_missing_metadata_fallback','Figures 8 and 10; original VP8 natural-video run uses fallback on every frame'],
        ['Synchronized dashboard and USB transport','dashboard.py\nDashboardHandler\nphone.py','test_dashboard_server.py\ntest_dashboard_streams_synchronized_decoded_preview\ntest_phone.py\ntest_phone_live_pipe_decodes_vectors_and_cleans_up','Figures 6 to 10 show actual local dashboard. Physical Android hardware remains untested.'],
    ],[1.1,1.45,2.45,2.1],size=9.1)
    rawpara('Controller tests and simulator race cases establish software behavior within their tested scope. They do not certify hardware atomicity, general model quality, or patent novelty. The Python prototype retains one final ViT attention block as its model-facing embodiment.')

    b.newpage();rawheading('8.9 Natural scene video validation')
    b.picture('natural-city-first-frame','Figure 11 Decoded natural scene used for the additional check',
        'An actual decoded frame from the H.264 derivative of Big City Life, showing pedestrians in an urban scene. This is input imagery, not a dashboard screenshot or a generated scene.',
        'Coverr, Big City Life, 2015, Wikimedia Commons, CC0 1.0, reference R14. Derived input and conversion provenance are supplied.',width=6)
    rawpara('We evaluated the complete original 1920 by 1080 VP8 file and a 640 by 360 H.264 derivative sampled at 12 fps with audio omitted. The derivative uses libx264, CRF 20, medium preset, GOP 48 and no B frames. The original SHA-1 matches the checksum listed on Commons; SHA-256 values and sampled source timestamps are recorded in natural-video-provenance.json. Both runs use the same final-block ViT, stored-K/V, reuse and baseline-comparison options as the controlled run.')
    v,h=natural['original-vp8'],natural['derived-h264']
    b.table(['Observation','Original VP8','H264 derivative'],[
        ['Frames ready / compared',f'{v["ready_frames"]} / {v["compared_frames"]}',f'{h["ready_frames"]} / {h["compared_frames"]}'],
        ['Metadata fallback',f'{v["metadata_fallback_rate"]*100:.2f}%',f'{h["metadata_fallback_rate"]*100:.2f}%'],
        ['Pages reused / recomputed',f'{v["kv_pages_reused"]} / {v["kv_pages_recomputed"]}',f'{h["kv_pages_reused"]} / {h["kv_pages_recomputed"]}'],
        ['Classifier top1 agreement','356 / 356', '170 / 171'],
        ['Mean absolute logit error',f'{v["mean_abs_logit_error"]:.8f}',f'{h["mean_abs_logit_error"]:.8f}'],
        ['Mean decoder plus processing ms',f'{v["mean_end_to_end_frame_ms"]:.2f}',f'{h["mean_end_to_end_frame_ms"]:.2f}'],
    ],[3.5,1.8,1.8])
    rawpara('The original decoded successfully with conservative fallback because no motion-vector side data was exposed. The H.264 derivative exposed motion evidence, but continuous scene change and exact feature validation produced zero reuse hits. Frame 124 at 10.25 seconds had one classifier disagreement, with mean absolute logit error 0.00412252. No causal ablation was performed to isolate which representation difference changed its ranking. This single convenience sample has no task labels or representativeness guarantee. It demonstrates compatibility and limitations, not an accuracy or reuse benefit across natural videos.')
    rawpara('The timing columns describe different encodings, resolutions and frame counts. They are observations from single runs, not a controlled codec comparison or an optimization speedup measurement.')
    b.newpage()

def heading(text):
    if text=='8.7 Verification and evidence traceability':
        capture_pages()
        text='8.10 Verification and evidence traceability'
    elif text=='8.8 Reproducing the demonstration':text='8.11 Reproducing the demonstration'
    return rawheading(text)

def emit(group,*args,**kwargs):
    rawemit(group,*args,**kwargs)
    if group==19:
        rawpara('R14  Coverr. Big City Life. 2015. Wikimedia Commons source file and license record. Listed as CC0 1.0 Universal; original VP8 duration 14.24 seconds, 1920 by 1080. Accessed 21 October 2026. Natural-video checks are reported in Section 8.9.')
        b.hyperlink(rawpara(''),'Big City Life source and license record',provenance['source_page'])

b.para=para;b.heading=heading;b.emit=emit
if __name__=='__main__':
    shutil.copyfile(ROOT/'output/evidence/natural-city-first-frame.png',b.ASSETS/'natural-city-first-frame.png')
    b.build_geometry();b.contents();b.package()
