# Magic Hour — Masked Skin Enhancer (FLUX.2 klein)

Internship project: build enhancer models that beat Magic Hour's current model and competitors (Remini, Topaz, Firefly), across three domains — **AI Skin**, **Photo**, **Portrait**.

**Core idea / moat:** use **face-parse masking** to preserve likeness in all cases, then swap the diffusion engine to **FLUX.2 [klein]**. The company's model ("Model A") relies on low-denoise instead of masking — so masking is our differentiator.

## Status

- AI Skin enhancer: working end-to-end with FLUX.2 klein + a mask that preserves likeness. Tunable intensity dial.
- Next: close the "humanness" gap vs Model A by adding a realism LoRA (Lenovo-UltraReal) and switching to low-denoise img2img.

## Repo layout

```
workflows/   ComfyUI workflow JSONs (the masked klein skin enhancer, versioned)
notebooks/   Colab launcher for ComfyUI
docs/        Technical plan + design notes
```

Findings live in Notion: *Magic Hour — AI Skin Enhancer Project* (Model A teardown, daily goals, work logs).

## Workflow changelog

| Version | What changed | Why |
|---|---|---|
| `..._KLEIN_skindetail.json` (v1) | Swapped SDXL/Juggernaut + UltimateSDUpscale for the FLUX.2 klein block; kept face-parse mask + composite-back. | Move to klein while preserving likeness. |
| `..._v2.json` | Geometry fix (klein renders at the crop's real size) + reframed prompt + **ImageBlend intensity dial** (blend_factor). | Fixed mask offset; added a single knob to avoid plastic-vs-lizard swings. |
| `..._v3.json` | Tighter mask: `blur_radius` 8→3, mask `expand` −4→−8, blend 0.35. | Removed gray cast bleeding onto clothing; cleaner mask edge. |
| `..._v4_LoRA.json` | Added `LoraLoaderModelOnly` (Lenovo-UltraReal, 0.8) on the klein model. | Port Model A's realism engine for more human skin. **Experimental** — verify LoRA loads on klein-4B. |

## Key technical notes

- The swap lives entirely inside the **"Add Skin Detail" subgraph**: klein stack (UNETLoader → CLIP/VAE loaders → SamplerCustomAdvanced → VAE Decode) replaced the SDXL stack. The face-detect → mask → composite-back → insert chain (likeness preservation) is untouched.
- klein has **no native denoise/strength slider** in the edit setup, which is why v2 added the ImageBlend dial. Model A instead uses a plain KSampler at **denoise 0.15–0.33** — the proper intensity control, and the planned next change.
- **License:** klein **4B = Apache 2.0** (commercial-safe). 9B and some realism LoRAs are non-commercial — fine for experiments, flag before shipping.

## Security

- **Never commit API keys.** CivitAI / HuggingFace / OpenAI keys must stay out of git (see `.gitignore`). Use Colab Secrets. Scrub keys from any notebook before committing.
- Keep this repo **private**.
