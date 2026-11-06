import os
from storyboard import generate_storyboard, shot_image_prompts
from frames import generate_frames
from consistency import report
from animate import write_outputs
from disclosure import burn_label, sidecar, label_text
from safety import check_prompt

def run(idea, technique="few-shot", seed=42, llm="auto", img="auto", outdir="outputs", music=True, fixed_seed=True):
    ok, hits = check_prompt(idea)
    if not ok: raise ValueError(f"Blocked by safety guardrail: {hits}")
    sb, llm_used, parsed = generate_storyboard(idea, technique, llm)
    prompts = shot_image_prompts(sb)
    for p in prompts:
        ok, hits = check_prompt(p)
        if not ok: raise ValueError(f"Generated shot blocked by guardrail: {hits}")
    frames, img_used = generate_frames(prompts, seed, img, fixed_seed)
    rows, summary = report(frames, prompts)
    models = f"LLM:{llm_used} img:{img_used}"
    labelled = [burn_label(f, models) for f in frames]
    out = write_outputs(labelled, outdir, music=music, label_comment=label_text(models))
    meta = sidecar(os.path.join(outdir, "clip.disclosure.json"), idea, models, seed, out["music_source"])
    return dict(storyboard=sb, prompts=prompts, frames=labelled, rows=rows, summary=summary, files=out, meta=meta, parsed=parsed)
