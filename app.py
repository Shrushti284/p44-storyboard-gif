"""Storyboard-to-GIF Animator: Gradio app (Hugging Face Spaces entry point)."""
import gradio as gr
from pipeline import run
from storyboard import build_prompt

def go(idea, technique, seed, llm, img, music):
    try:
        r = run(idea, technique, int(seed), llm, img, "outputs", music)
    except ValueError as e:
        raise gr.Error(str(e))
    sb = r["storyboard"]
    shots = [[s["shot"], s["action"], s["camera"], p] for s, p in zip(sb["shots"], r["prompts"])]
    cons = [[x["shot"], x["caption"], None if x["caption_vs_prompt"] != x["caption_vs_prompt"] else round(x["caption_vs_prompt"], 3),
             None if x["adjacent_hist_corr"] != x["adjacent_hist_corr"] else round(x["adjacent_hist_corr"], 3)] for x in r["rows"]]
    note = (f"**AI-generated.** Models: {r['meta']['models']} · seed {seed} · music: {r['meta']['music']} · "
            f"storyboard JSON parsed from LLM: {r['parsed']} · subject-colour similarity {r['summary']['mean_subject_colour_sim']:.3f}")
    return shots, r["frames"], cons, r["files"]["gif"], r["files"]["mp4"], note

with gr.Blocks(title="Storyboard-to-GIF Animator") as demo:
    gr.Markdown("# Storyboard-to-GIF Animator\nIdea → 6-shot storyboard (LLM) → frames (fixed seed) → cross-fade GIF/MP4 + music → AI-generated label.\n"
                "> Every output is labelled **AI-generated** (burned into the frames, in MP4 metadata and a sidecar JSON).")
    with gr.Tab("Create"):
        idea = gr.Textbox(label="Idea", value="a tiny fox finds a lantern in a snowy forest")
        with gr.Row():
            technique = gr.Radio(["few-shot", "zero-shot"], value="few-shot", label="Prompt technique (U1)")
            seed = gr.Number(42, label="Fixed seed (U3)", precision=0)
            llm = gr.Dropdown(["auto", "gemini", "groq", "local", "template"], value="auto", label="LLM backend")
            img = gr.Dropdown(["auto", "sd-turbo", "hf-api", "procedural"], value="auto", label="Image backend")
            music = gr.Checkbox(True, label="Music")
        btn = gr.Button("Generate", variant="primary"); note = gr.Markdown()
        shots = gr.Dataframe(headers=["shot", "action", "camera", "image prompt"], label="Shot list (U2)")
        gal = gr.Gallery(label="Frames (U3)", columns=6)
        cons = gr.Dataframe(headers=["shot", "BLIP caption", "caption~prompt", "next-frame hist corr"], label="Consistency check (BLIP)")
        with gr.Row(): gif = gr.Image(label="GIF"); mp4 = gr.Video(label="MP4 with music")
        btn.click(go, [idea, technique, seed, llm, img, music], [shots, gal, cons, gif, mp4, note])
    with gr.Tab("Prompts (U1)"):
        gr.Code(build_prompt("<your idea>", "zero-shot"), label="Zero-shot"); gr.Code(build_prompt("<your idea>", "few-shot"), label="Few-shot")
    with gr.Tab("Responsible AI"):
        gr.Markdown(open("MODEL_CARD.md").read())
demo.launch() if __name__ == "__main__" else None
