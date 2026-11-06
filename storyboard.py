"""U1 + U2: shot-list prompts and LLM storyboard generation (6 shots)."""
import json, os, re

N_SHOTS = 6

# ---- U1: two prompt techniques, compared in eval.py ----
ZERO_SHOT = (
    "You are a storyboard artist. Turn this idea into exactly {n} shots for a short animated clip.\n"
    "Idea: {idea}\n"
    'Reply with ONLY JSON: {{"character": "...", "style": "...", "shots": [{{"shot": 1, "action": "...", "camera": "...", "image_prompt": "..."}}]}}'
)
FEW_SHOT = (
    "You are a storyboard artist. Output exactly {n} shots as JSON. Keep the SAME character and style in every shot.\n"
    "Example idea: a robot learns to paint\n"
    'Example output: {{"character": "small yellow robot with round eyes", "style": "flat watercolor, soft light", '
    '"shots": [{{"shot": 1, "action": "robot finds a brush", "camera": "wide", "image_prompt": "small yellow robot with round eyes holding a brush in a sunny studio"}}, '
    '{{"shot": 2, "action": "first messy stroke", "camera": "close-up", "image_prompt": "small yellow robot with round eyes making a blue stroke on canvas"}}]}}\n'
    "Now the real task. Idea: {idea}\nOutput ONLY JSON with exactly {n} shots."
)
TECHNIQUES = {"zero-shot": ZERO_SHOT, "few-shot": FEW_SHOT}

def build_prompt(idea, technique="few-shot"):
    return TECHNIQUES[technique].format(idea=idea.strip(), n=N_SHOTS)

# ---- LLM backends ----
def _call_gemini(prompt):
    import requests
    key = os.environ["GEMINI_API_KEY"]
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={key}"
    r = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=60)
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]

def _call_groq(prompt):
    import requests
    r = requests.post("https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": "Bearer " + os.environ["GROQ_API_KEY"]},
        json={"model": "llama-3.1-8b-instant", "messages": [{"role": "user", "content": prompt}], "temperature": 0.7},
        timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def _call_local(prompt):  # open model fallback (Qwen2.5-1.5B-Instruct)
    from transformers import pipeline
    pipe = pipeline("text-generation", model="Qwen/Qwen2.5-1.5B-Instruct")
    out = pipe([{"role": "user", "content": prompt}], max_new_tokens=700)
    return out[0]["generated_text"][-1]["content"]

def _template(idea):
    """Deterministic offline fallback so the app never dies without a key."""
    beats = [("establishing the world", "wide"), ("hero appears", "medium"), ("a problem arises", "close-up"),
             ("hero reacts", "close-up"), ("the turning point", "medium"), ("resolution at sunset", "wide")]
    char = "the main character from the idea"
    return {"character": char, "style": "soft storybook illustration",
            "shots": [{"shot": i + 1, "action": a, "camera": c, "image_prompt": f"{idea}, {a}"} for i, (a, c) in enumerate(beats)]}

def parse_storyboard(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m: raise ValueError("no JSON found")
    data = json.loads(m.group(0))
    shots = data["shots"]
    if len(shots) != N_SHOTS: raise ValueError(f"expected {N_SHOTS} shots, got {len(shots)}")
    for i, s in enumerate(shots): s["shot"] = i + 1
    data.setdefault("character", ""); data.setdefault("style", "illustration")
    return data

def generate_storyboard(idea, technique="few-shot", backend="auto"):
    """Returns (storyboard dict, backend_used, parse_ok)."""
    prompt = build_prompt(idea, technique)
    order = {"auto": ["gemini", "groq", "local"], "gemini": ["gemini"], "groq": ["groq"], "local": ["local"], "template": []}[backend]
    fns = {"gemini": (_call_gemini, "GEMINI_API_KEY"), "groq": (_call_groq, "GROQ_API_KEY"), "local": (_call_local, None)}
    for name in order:
        fn, env = fns[name]
        if env and not os.environ.get(env): continue
        try:
            return parse_storyboard(fn(prompt)), name, True
        except Exception as e:
            print(f"[storyboard] {name} failed: {e}")
    return _template(idea), "template", False

def shot_image_prompts(sb):
    """Character + style anchor is prepended to every shot: first lever for cross-frame consistency."""
    anchor = f"{sb['character']}, {sb['style']}".strip(", ")
    return [f"{anchor}. {s['image_prompt']}" for s in sb["shots"]]
