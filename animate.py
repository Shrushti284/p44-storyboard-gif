"""U4: cross-fade frames to GIF + MP4, with music (MusicGen if available, else synthesised)."""
import os, subprocess, wave
import numpy as np
from PIL import Image
import imageio.v2 as imageio
import imageio_ffmpeg

FPS = 12

def crossfade(frames, hold=1.0, fade=0.6, fps=FPS):
    out = []
    for i, f in enumerate(frames):
        out += [np.array(f.convert("RGB"))] * int(hold * fps)
        if i + 1 < len(frames):
            a, b = np.array(f.convert("RGB")).astype(np.float32), np.array(frames[i + 1].convert("RGB")).astype(np.float32)
            n = max(1, int(fade * fps))
            for k in range(1, n + 1):
                t = k / (n + 1); out.append(((1 - t) * a + t * b).astype(np.uint8))
    return out

def synth_music(seconds, path, mood="calm", sr=22050):
    """Procedural pentatonic arpeggio + pad. Fallback when MusicGen is unavailable."""
    scale = {"calm": [0, 3, 5, 7, 10], "bright": [0, 2, 4, 7, 9]}.get(mood, [0, 2, 4, 7, 9])
    t = np.arange(int(seconds * sr)) / sr; y = np.zeros_like(t); step = 0.4
    for k in range(int(seconds / step)):
        f = 220 * 2 ** ((scale[(k * 2) % 5] + 12 * ((k // 5) % 2)) / 12)
        s, e = int(k * step * sr), min(len(t), int((k + 1) * step * sr)); tt = t[s:e] - t[s]
        y[s:e] += 0.35 * np.sin(2 * np.pi * f * tt) * np.exp(-4 * tt)
    y += 0.12 * np.sin(2 * np.pi * 110 * t)
    y *= np.minimum(1, np.minimum(t, seconds - t) / 1.0)
    pcm = (np.clip(y / max(1e-6, np.abs(y).max()), -1, 1) * 0.8 * 32767).astype(np.int16)
    with wave.open(path, "wb") as w: w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(pcm.tobytes())
    return "synth"

def make_music(seconds, path, prompt="gentle cinematic ambient"):
    try:
        import torch, scipy.io.wavfile as wf
        from transformers import pipeline
        syn = pipeline("text-to-audio", "facebook/musicgen-small")
        out = syn(prompt, forward_params={"max_new_tokens": min(1500, int(seconds * 50))})
        wf.write(path, out["sampling_rate"], np.squeeze(out["audio"]).astype(np.float32)); return "musicgen-small"
    except Exception as e:
        print("[music] MusicGen unavailable:", type(e).__name__); return synth_music(seconds, path)

def write_outputs(frames, outdir, name="clip", music=True, mood_prompt="gentle cinematic ambient", label_comment=""):
    os.makedirs(outdir, exist_ok=True)
    seq = crossfade(frames); secs = len(seq) / FPS
    gif = os.path.join(outdir, name + ".gif")
    imageio.mimsave(gif, [Image.fromarray(a).quantize(128) .convert("RGB") for a in seq][::1], duration=1000 / FPS, loop=0)
    silent = os.path.join(outdir, name + "_silent.mp4")
    imageio.mimsave(silent, seq, fps=FPS, codec="libx264", macro_block_size=1, quality=7)
    mp4, src = silent, "none"
    if music:
        wav = os.path.join(outdir, name + ".wav"); src = make_music(secs, wav, mood_prompt)
        mp4 = os.path.join(outdir, name + ".mp4")
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", silent, "-i", wav, "-c:v", "copy", "-c:a", "aac", "-shortest",
                        "-metadata", f"comment={label_comment}", "-metadata", "title=AI-generated storyboard clip", mp4],
                       check=True, capture_output=True)
    return {"gif": gif, "mp4": mp4, "seconds": secs, "music_source": src}
