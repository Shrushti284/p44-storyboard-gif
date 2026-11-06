"""U5: input guardrail."""
BLOCK = ["nude", "naked", "porn", "gore", "beheading", "child abuse", "suicide method", "bomb making", "how to make a bomb",
         "deepfake of", "terror attack"]
def check_prompt(text):
    t = text.lower()
    hits = [w for w in BLOCK if w in t]
    return (len(hits) == 0, hits)
