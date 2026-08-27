#!/usr/bin/env python
"""Generate one sample grid per configuration, annotated with its own metrics.

Samples are only useful if you can tell which number produced which image, so each
is saved with its SSIM against the same-seed exact output and its ImageReward. The
reference column is the full computation at the model's own step count.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from dit_residual_delta.config import load_config
from dit_residual_delta.metrics import image_tensor, lpips_alex, skimage_ssim
from dit_residual_delta.pipeline import load_pixart_pipeline, load_prompt_embeddings
from dit_residual_delta.surrogate import load_surrogate_bank
from dit_residual_delta.variants import build_variant, runtime_for_variant

_B={}
def bank(spec):
    p=spec.get("surrogate_checkpoint")
    if p is None: return None
    if p not in _B:
        _B[p]=load_surrogate_bank(p, dtype=(torch.float16 if a.surrogate_dtype == "fp16"
                                            else torch.float32))
    return _B[p]

ap=argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--prompt-file", required=True)
ap.add_argument("--embeddings", required=True)
ap.add_argument("--seed", type=int, default=9401)
ap.add_argument("--limit", type=int, default=5)
ap.add_argument("--variants-file", required=True)
ap.add_argument("--output-dir", required=True)
ap.add_argument("--surrogate-dtype", choices=["fp16", "fp32"], default="fp16",
                help="deploy precision. fp16 matches every earlier result, but a "
                     "corrector trained to convergence can overflow it -- m_k7 "
                     "returns inf on 111 of 112 calls and the image is all NaN")
a=ap.parse_args()

import ImageReward as RM
reward=RM.load("ImageReward-v1.0", device="cuda")
cfg=load_config(a.config)["model"]; ref_steps=int(cfg["num_inference_steps"])
common={"height":int(cfg["height"]),"width":int(cfg["width"]),
        "guidance_scale":float(cfg["guidance_scale"])}
pipe=load_pixart_pipeline(with_text_encoder=False).to("cuda")
prompts=[l.strip() for l in Path(a.prompt_file).read_text().splitlines() if l.strip()][:a.limit]
emb=load_prompt_embeddings(a.embeddings)
specs=json.loads(Path(a.variants_file).read_text())
out=Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)

def gen(embed, steps, spec=None):
    kw={**common, **embed, "num_inference_steps": steps}
    g=torch.Generator(device="cuda").manual_seed(a.seed)
    with torch.inference_mode():
        if spec is None: return pipe(**kw, generator=g).images[0]
        v=build_variant(spec, steps, len(pipe.transformer.transformer_blocks))
        with runtime_for_variant(pipe.transformer, v, bank(spec)):
            return pipe(**kw, generator=g).images[0]

meta={}
for pi,p in enumerate(prompts):
    ref=gen(emb[p], ref_steps)
    rp=out/f"p{pi}_reference.png"; ref.save(rp)
    meta[f"p{pi}"]={"prompt":p,"reference":{"ssim":1.0,"reward":reward.score(p,str(rp))}}
    for spec in specs:
        img=gen(emb[p], int(spec["steps"]), None if spec.get("exact") else spec)
        ip=out/f"p{pi}_{spec['name']}.png"; img.save(ip)
        meta[f"p{pi}"][spec["name"]]={"ssim":skimage_ssim(img,ref),
                                      "reward":reward.score(p,str(ip))}
    print(json.dumps({"prompt":pi,"done":True}), flush=True)
(out/"meta.json").write_text(json.dumps(meta, indent=1))
print(json.dumps({"written":str(out/"meta.json"),"prompts":len(prompts)}))
