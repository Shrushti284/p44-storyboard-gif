# P44 Storyboard-to-GIF Animator

Idea → 6-shot storyboard (LLM) → frames with a fixed seed → BLIP consistency check → cross-fade GIF/MP4 with music → "AI-generated" label.

| Unit | Task | File |
|---|---|---|
| U1 | Shot-list prompts (zero-shot vs few-shot) | `storyboard.py` |
| U2 | LLM storyboard + JSON validation | `storyboard.py` |
| U3 | Fixed-seed frames (sd-turbo), BLIP + MiniLM checks | `frames.py`, `consistency.py` |
| U4 | Cross-fade, GIF/MP4, music | `animate.py` |
| U5 | Disclosure label, guardrail, model card | `disclosure.py`, `safety.py`, `MODEL_CARD.md` |

## Run
```
pip install -r requirements.txt
export GEMINI_API_KEY=...   # or GROQ_API_KEY; on Spaces use Settings > Secrets
python app.py
python eval.py auto auto    # writes results.json (re-run with real models for the report)
```
Backends degrade gracefully: LLM → template, image → procedural, music → synthesised. The footer note of every run states which backend was used.

## Deploy (Hugging Face Spaces)
Create a Gradio Space, upload these files, add secrets `GEMINI_API_KEY` / `HF_TOKEN`. For free CPU hosts set image backend to `hf-api`.
