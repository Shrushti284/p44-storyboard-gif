"""U3b: consistency checks. BLIP captions each frame; a BERT-family sentence model scores caption vs prompt;
adjacent-frame similarity is measured with colour histograms (cv2) and, when available, CLIP-free embeddings."""
import re
import numpy as np

_BLIP = None
def caption(img):
    global _BLIP
    try:
        if _BLIP is None:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            _BLIP = (BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base"),
                     BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base"))
        proc, model = _BLIP
        out = model.generate(**proc(img, return_tensors="pt"), max_new_tokens=30)
        return proc.decode(out[0], skip_special_tokens=True), "blip"
    except Exception:
        return "(BLIP unavailable)", "none"

_ST = None
def text_sim(a, b):
    """BERT-family cosine (all-MiniLM-L6-v2) if installed, else token Jaccard."""
    global _ST
    try:
        if _ST is None:
            from sentence_transformers import SentenceTransformer
            _ST = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        e = _ST.encode([a, b], normalize_embeddings=True); return float(e[0] @ e[1]), "minilm"
    except Exception:
        t = lambda s: set(re.findall(r"[a-z]+", s.lower()))
        x, y = t(a), t(b); return (len(x & y) / max(1, len(x | y))), "jaccard"

def hist_sim(a, b):
    import cv2
    h = []
    for im in (a, b):
        arr = cv2.cvtColor(np.array(im.convert("RGB")), cv2.COLOR_RGB2HSV)
        x = cv2.calcHist([arr], [0, 1], None, [16, 16], [0, 180, 0, 256]); cv2.normalize(x, x); h.append(x)
    return float(cv2.compareHist(h[0], h[1], cv2.HISTCMP_CORREL))

def hero_colour_sim(a, b):
    """Cheap subject check: mean colour of the central region."""
    f = lambda im: np.array(im.convert("RGB").crop((150, 270, 380, 420))).reshape(-1, 3).mean(0)
    d = np.linalg.norm(f(a) - f(b)); return float(1 - d / 441.7)

def report(frames, prompts):
    rows = []
    for i, (im, p) in enumerate(zip(frames, prompts)):
        cap, cb = caption(im); sim, sb = text_sim(cap, p) if cb == "blip" else (float("nan"), "n/a")
        nxt = hist_sim(im, frames[i + 1]) if i + 1 < len(frames) else float("nan")
        rows.append({"shot": i + 1, "caption": cap, "caption_vs_prompt": sim, "adjacent_hist_corr": nxt, "captioner": cb, "text_sim": sb})
    adj = [r["adjacent_hist_corr"] for r in rows if r["adjacent_hist_corr"] == r["adjacent_hist_corr"]]
    hero = [hero_colour_sim(frames[i], frames[i + 1]) for i in range(len(frames) - 1)]
    return rows, {"mean_adjacent_hist_corr": float(np.mean(adj)), "mean_subject_colour_sim": float(np.mean(hero))}
