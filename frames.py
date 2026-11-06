"""U3a: frame generation with a FIXED seed (same seed for every shot)."""
import hashlib, os, random
from PIL import Image, ImageDraw, ImageFont

SIZE = 512

def _font(sz):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "DejaVuSans.ttf"]:
        try: return ImageFont.truetype(p, sz)
        except Exception: pass
    return ImageFont.load_default()

_PIPE = None
def _sd_turbo():
    global _PIPE
    if _PIPE is None:
        import torch
        from diffusers import AutoPipelineForText2Image
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _PIPE = AutoPipelineForText2Image.from_pretrained("stabilityai/sd-turbo",
                    torch_dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev)
    return _PIPE

def gen_sd_turbo(prompts, seed):
    import torch
    pipe = _sd_turbo(); imgs = []
    for p in prompts:
        g = torch.Generator(device=pipe.device).manual_seed(seed)  # fixed seed for every shot
        imgs.append(pipe(p, num_inference_steps=2, guidance_scale=0.0, generator=g, height=SIZE, width=SIZE).images[0])
    return imgs

def gen_hf_api(prompts, seed):
    import io, requests
    h = {"Authorization": "Bearer " + os.environ["HF_TOKEN"]}
    imgs = []
    for p in prompts:
        r = requests.post("https://api-inference.huggingface.co/models/stabilityai/sd-turbo", headers=h,
                          json={"inputs": p, "parameters": {"seed": seed}}, timeout=120)
        r.raise_for_status(); imgs.append(Image.open(io.BytesIO(r.content)).convert("RGB").resize((SIZE, SIZE)))
    return imgs

def gen_procedural(prompts, seed, fixed=True):
    """Offline stand-in: a consistent hero on a scene whose palette derives from the prompt.
    NOT a diffusion model. Used for tests and as the last-resort fallback."""
    imgs = []
    for i, p in enumerate(prompts):
        s = seed if fixed else random.randrange(10**6)
        rnd = random.Random(s)                        # hero look depends only on the seed
        hero = tuple(rnd.randrange(80, 255) for _ in range(3))
        hx = rnd.randrange(150, 360)
        rs = random.Random(int(hashlib.md5(p.encode()).hexdigest(), 16) % 10**8)  # scene depends on prompt
        top = tuple(rs.randrange(60, 230) for _ in range(3)); bot = tuple(rs.randrange(20, 120) for _ in range(3))
        im = Image.new("RGB", (SIZE, SIZE)); d = ImageDraw.Draw(im)
        for y in range(SIZE):
            t = y / SIZE; d.line([(0, y), (SIZE, y)], fill=tuple(int(top[k] * (1 - t) + bot[k] * t) for k in range(3)))
        d.rectangle([0, 380, SIZE, SIZE], fill=tuple(c // 2 for c in bot))
        cx = hx + (i - 2) * 12
        d.ellipse([cx - 40, 270, cx + 40, 350], fill=hero); d.rectangle([cx - 30, 340, cx + 30, 420], fill=hero)
        d.ellipse([cx - 22, 295, cx - 8, 309], fill="white"); d.ellipse([cx + 8, 295, cx + 22, 309], fill="white")
        d.text((14, 12), f"Shot {i+1}", fill="white", font=_font(22))
        imgs.append(im)
    return imgs

def generate_frames(prompts, seed=42, backend="auto", fixed=True):
    """Returns (images, backend_used)."""
    order = [backend] if backend != "auto" else ["sd-turbo", "hf-api", "procedural"]
    for b in order:
        try:
            if b == "sd-turbo": return gen_sd_turbo(prompts, seed if fixed else random.randrange(10**6)), b
            if b == "hf-api" and os.environ.get("HF_TOKEN"): return gen_hf_api(prompts, seed), b
            if b == "procedural": return gen_procedural(prompts, seed, fixed), b
        except Exception as e:
            print(f"[frames] {b} failed: {e}")
    return gen_procedural(prompts, seed, fixed), "procedural"
