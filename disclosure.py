"""U5: AI-generated disclosure label: visible burn-in + metadata + sidecar JSON."""
import json, datetime
from PIL import ImageDraw
from frames import _font

LABEL = "AI-generated"
def label_text(models):
    return f"{LABEL} · {models}"

def burn_label(img, models):
    im = img.copy(); d = ImageDraw.Draw(im); t = label_text(models); f = _font(15)
    w = d.textlength(t, font=f); x, y = im.width - w - 18, im.height - 30
    d.rectangle([x - 6, y - 4, x + w + 6, y + 22], fill=(0, 0, 0)); d.text((x, y), t, fill="white", font=f)
    return im

def sidecar(path, idea, models, seed, music_src):
    meta = {"label": LABEL, "idea": idea, "models": models, "seed": seed, "music": music_src,
            "created_utc": datetime.datetime.utcnow().isoformat() + "Z",
            "note": "Synthetic media. Not a record of real events. Licences: see MODEL_CARD.md."}
    with open(path, "w") as f: json.dump(meta, f, indent=2)
    return meta
