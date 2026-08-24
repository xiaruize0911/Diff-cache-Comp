#!/usr/bin/env python
"""Build PixArt-format T5 embeddings for COCO captions.

PixArt-Sigma conditions on T5-XXL (d_model 4096, 300 tokens). The staged PixArt
snapshot kept only the tokenizer/config, not the encoder weights, but FLUX ships the
same base encoder (google/t5-v1_1-xxl), so we load it from there. Every variant in a
comparison uses the identical embeddings, so any residual preprocessing difference
relative to the original PixArt pipeline shifts all variants equally and cannot bias
a variant-vs-variant comparison.
"""
from __future__ import annotations
import argparse, json, zipfile
from pathlib import Path
import torch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--coco-zip", required=True)
    p.add_argument("--num", type=int, default=2000)
    p.add_argument("--t5", default="/workspace/models/flux1_dev/text_encoder_2")
    p.add_argument("--tokenizer", default="/workspace/models/flux1_dev/tokenizer_2")
    p.add_argument("--max-length", type=int, default=300)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--output", required=True)
    p.add_argument("--prompt-list-out", default=None)
    args = p.parse_args()

    from transformers import T5EncoderModel, AutoTokenizer
    z = zipfile.ZipFile(args.coco_zip)
    name = next(n for n in z.namelist() if "captions_val2017" in n)
    anns = json.loads(z.read(name))["annotations"]
    # one caption per image id, deterministic order -> distinct scenes, no duplicates
    seen, caps = set(), []
    for a in sorted(anns, key=lambda x: (x["image_id"], x["id"])):
        if a["image_id"] in seen:
            continue
        seen.add(a["image_id"])
        caps.append(" ".join(a["caption"].strip().split()))
        if len(caps) >= args.num:
            break
    print(json.dumps({"captions": len(caps), "distinct_images": len(seen)}), flush=True)

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    enc = T5EncoderModel.from_pretrained(args.t5, torch_dtype=torch.float16).to("cuda").eval()

    def embed(texts):
        b = tok(texts, padding="max_length", max_length=args.max_length,
                truncation=True, add_special_tokens=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = enc(input_ids=b.input_ids, attention_mask=b.attention_mask)[0]
        return out.cpu().half(), b.attention_mask.cpu().long()

    store = {"prompts": {}, "source": f"coco_val2017:{args.num}"}
    neg_e, neg_m = embed([""])
    store["negative"] = {"prompt_embeds": neg_e, "prompt_attention_mask": neg_m}
    for i in range(0, len(caps), args.batch):
        chunk = caps[i : i + args.batch]
        e, m = embed(chunk)
        for j, c in enumerate(chunk):
            store["prompts"][c] = {"prompt_embeds": e[j : j + 1],
                                   "prompt_attention_mask": m[j : j + 1]}
        if (i // args.batch) % 20 == 0:
            print(json.dumps({"done": min(i + args.batch, len(caps)), "of": len(caps)}), flush=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(store, args.output)
    if args.prompt_list_out:
        Path(args.prompt_list_out).write_text("\n".join(caps) + "\n")
    print(json.dumps({"saved": args.output, "n": len(store["prompts"])}))


if __name__ == "__main__":
    main()
