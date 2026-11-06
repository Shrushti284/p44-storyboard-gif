## Model card: Storyboard-to-GIF Animator

**Intended use:** short, illustrative animated clips from a text idea (education, storytelling). Not for realistic depictions of real people or events.

| Stage | Model | Licence / note |
|---|---|---|
| Storyboard LLM | Gemini or Groq free-tier API; or Qwen2.5-1.5B-Instruct (Apache-2.0) | API terms of the provider; keys stored in Space secrets |
| Image generation | stabilityai/sd-turbo | Stability AI Community Licence (check commercial limits) |
| Caption check | Salesforce/blip-image-captioning-base (BSD-3) | |
| Text similarity (BERT-family) | all-MiniLM-L6-v2 (Apache-2.0) | |
| Music | facebook/musicgen-small (**CC-BY-NC 4.0, non-commercial**) | synthesised fallback otherwise |

**Responsible-AI measures**
- Disclosure: "AI-generated" label burned into every frame, MP4 metadata and a `clip.disclosure.json` sidecar (models, seed, time).
- Guardrail: keyword blocklist applied to the user idea *and* every generated shot prompt before image generation.
- Hallucination/consistency test: BLIP caption vs shot prompt similarity per frame, plus adjacent-frame similarity.

**Known limitations / failure cases**
- A fixed seed keeps layout/palette similar, not character identity; the hero can still change between shots when prompts differ much.
- The keyword guardrail misses paraphrases and can over-block innocent text.
- Diffusion models carry dataset biases (e.g. default skin tone, gender, setting); no person-specific likeness should be requested.
- Small LLMs sometimes return invalid JSON; the app falls back to a template storyboard and reports this.
