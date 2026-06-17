# AI Skin Enhancer — Masked FLUX.2 [klein] Workflow Plan

**Author:** Vamika · **Date:** 2026-06-12 · **Status:** Design + build recipe (not yet load-tested in ComfyUI)

This document does three things your boss asked for: (1) explains *exactly* how the reference workflow preserves likeness with a mask, (2) lays out how to swap the slow SDXL/Juggernaut stage for FLUX.2 [klein], and (3) gives a node-by-node recipe to build it in the Comfy GUI.

One honest caveat up front: I cannot run ComfyUI or a GPU, so I could not load-test the resulting graph. Everything below is grounded in the actual node graph of your reference file and the official ComfyUI FLUX.2 klein templates, but the final wiring must be verified in your Colab GUI. I flag the parts I'm less than certain about.

---

## 1. How the reference workflow actually preserves likeness

The file `templates_hellorob_facegen_skindetail_upscale.json` is built from three sub-graphs:

| Stage | Subgraph | What it does |
|---|---|---|
| Source | *Text to Image (Z-Image-Turbo)* + `LoadImage` + `Image Switch` | Either loads your photo or generates one. The switch picks which. |
| **Enhance** | *Add Skin Detail (SDXL)* (40 nodes) | The core. Detects the face, builds a feature mask, enhances skin, composites back. |
| Upscale | *Image Upscale (SeedVR2)* | Final high-quality upscale with the SeedVR2 7B model. This is the step your boss said "adds a lot of value." |

The likeness-preservation trick lives entirely in the **Enhance** subgraph, and it is *not* "edit the face gently." It is **enhance the whole face, then paste only the skin back through a mask**:

1. **Detect & crop the face** — `face_yolov8m.pt` finds the face; the workflow crops just that region (faster, higher detail).
2. **Segment facial features** — a FaceParsing model (BiSeNet-style) labels every region: skin, nose, eyes, brows, lips, etc. `FaceParsingResultsParser` turns selected labels into a mask. Its boolean toggles are what the note exposes as `nose, r_eye, l_eye, r_brow, l_brow, u_lip, l_lip` — **features toggled OFF are excluded from the mask, so they are kept as the original pixels.**
3. **Feather the mask** — `GrowMaskWithBlur` (blur ≈ 8px) softens the edge so the paste blends seamlessly.
4. **Enhance the crop** — `UltimateSDUpscale` runs a low-denoise (0.2) img2img pass on the *whole* cropped face using Juggernaut-XL + two skin LoRAs (`better_freckles`, `skin_realism_acne...`). This is where new skin detail/pores come from.
5. **Composite back through the mask** — `ImageCompositeMasked` lays the enhanced crop over the *original* crop, but **only where the mask is white (skin).** Eyes, brows, and lips revert to the untouched original → identity is preserved.
6. **Paste back into the full image** — `ImageInsertWithBBox` drops the corrected crop back where the face was.

**The key realization:** likeness is protected by step 5 (the mask composite), not by the diffusion model. That machinery is completely model-agnostic — which is exactly why we can swap the model underneath it without losing the eye-preservation behavior.

---

## 2. The catch: FLUX.2 klein edits differently than SDXL

This is the most important thing to understand before building, and it changes one of the user controls.

SDXL img2img (what the workflow uses now) has a **denoise strength** knob — 0.15–0.35 in the note — that directly controls "how much to change." FLUX.2 klein's editing does **not** work that way. In ComfyUI, klein editing is **instruction + reference-latent based**:

- The input image is `VAEEncode`d and fed through a `ReferenceLatent` node as conditioning, alongside a natural-language instruction prompt (e.g. *"add realistic skin texture and visible pores"*).
- Sampling is `RandomNoise → KSamplerSelect (euler) → Flux2Scheduler → CFGGuider (CFG 1.0) → SamplerCustomAdvanced`. Distilled 4B runs in **4 steps**; base in ~20.
- There is **no denoise slider.** Change magnitude is governed by the instruction wording and by — critically — **the mask composite in step 5.**

**Consequence:** with klein, the feature mask isn't optional polish anymore; it becomes the *primary* likeness safeguard, because klein regenerates rather than lightly denoising. Good news: that mask machinery already exists and we keep it as-is. We just need a reasonably tight skin mask and the feathering you already have.

I'm moderately confident, not certain, that the official "edit" pattern uses `EmptyFlux2LatentImage` as the sampling canvas while the original image rides in through `ReferenceLatent`. An alternative worth A/B testing is feeding the `VAEEncode` latent of the crop directly as the sampler's latent (more like classic img2img). Test both; the latter may retain structure better for a subtle skin pass. **Verify in the GUI.**

---

## 3. The swap design

Keep the entire pipeline. Replace only the inner enhance engine inside the *Add Skin Detail* subgraph.

```
Cropped face (Reroute 105)
        │
        ▼
┌──────────────────────────────┐        ┌─────────────────────────────┐
│  OLD (delete):               │        │  NEW (add): FLUX.2 klein     │
│  UltimateSDUpscale (76)      │   ==>  │  edit block                  │
│  + Juggernaut checkpoint(116)│        │  UNETLoader(klein 4b)        │
│  + 2 SDXL LoRAs (227,122)    │        │  CLIPLoader(qwen_3_4b)       │
│  + Pos/Neg prompts (78,117)  │        │  VAELoader(flux2-vae)        │
│  + UpscaleModelLoader(72)    │        │  VAEEncode → ReferenceLatent │
│  + Upscale By (98)           │        │  CLIPTextEncode(instruction) │
│  + model/vae/clip reroutes   │        │  RandomNoise/KSamplerSelect/ │
│                              │        │  Flux2Scheduler/CFGGuider/   │
│                              │        │  SamplerCustomAdvanced/VAEDec│
└──────────────────────────────┘        └─────────────────────────────┘
        │                                          │
        ▼ (enhanced face image)                    ▼
ImageResizeKJv2 (196)  ── resize back to crop size ──┘
        ▼
ImageCompositeMasked (176)  ← GrowMaskWithBlur (179) ← FaceParse mask   [KEEP — preserves eyes]
        ▼
ImageInsertWithBBox (180)  → back into full image                       [KEEP]
        ▼
Image Upscale (SeedVR2)                                                  [KEEP]
```

**Cut point in / out:** the cropped face arrives at `Reroute [105]`; the enhanced face must exit into `ImageResizeKJv2 [196]`. Wire the klein block between exactly those two points and nothing else in the subgraph needs to change.

---

## 4. Build recipe (do this in the Comfy GUI)

Your boss expects GUI work here, and it's the reliable path. Work *inside* the *Add Skin Detail (SDXL)* subgraph (double-click it to enter).

**Step A — delete the SDXL engine.** Delete these nodes: `UltimateSDUpscale` (76), `CheckpointLoaderSimple` (116), both `LoraLoaderModelOnly` (122, 227), `Positive Prompt` (78), `Negative Prompt` (117), `UpscaleModelLoader` (72), `Upscale By` (98), and the now-dangling model/clip/vae reroutes (99–104). **Do not touch** the FaceParsing nodes, the mask chain, `ImageResizeKJv2` (196), `ImageCompositeMasked` (176), or `ImageInsertWithBBox` (180).

**Step B — add the klein edit block** and wire it:

| New node | Key setting | Wires to |
|---|---|---|
| `UNETLoader` | `flux-2-klein-4b.safetensors` (distilled) | → `CFGGuider.model` |
| `CLIPLoader` | `qwen_3_4b.safetensors`, type `flux2` | → `CLIPTextEncode.clip` |
| `VAELoader` | `flux2-vae.safetensors` | → `VAEEncode.vae` and `VAEDecode.vae` |
| `VAEEncode` | — | image ← `Reroute [105]` (cropped face); LATENT → `ReferenceLatent` |
| `CLIPTextEncode` | the skin instruction (below) | → `ReferenceLatent.conditioning` |
| `ReferenceLatent` | — | → `CFGGuider.positive` |
| `ConditioningZeroOut` | (fed from the same CLIPTextEncode) | → `CFGGuider.negative` |
| `EmptyFlux2LatentImage` | 1024×1024 (multiple of 16) | → `SamplerCustomAdvanced.latent_image` |
| `RandomNoise` | randomize | → `SamplerCustomAdvanced.noise` |
| `KSamplerSelect` | `euler` | → `SamplerCustomAdvanced.sampler` |
| `Flux2Scheduler` | steps **4**, w/h 1024 | → `SamplerCustomAdvanced.sigmas` |
| `CFGGuider` | CFG **1.0** | → `SamplerCustomAdvanced.guider` |
| `SamplerCustomAdvanced` | — | LATENT → `VAEDecode` |
| `VAEDecode` | — | IMAGE → `ImageResizeKJv2 [196]` |

`ImageResizeKJv2 [196]` already resizes back to the original crop dimensions (via `GetImageSize [193]`), so klein can generate at 1024 and it will be fit back correctly.

**Step C — the instruction prompt.** klein wants natural-language instructions, not SDXL tag-soup. Start with something like:

> *"Enhance only the skin: add realistic fine skin texture, natural pores, and subtle blemishes. Keep the exact same person, identity, facial structure, eyes, and expression. Photorealistic, no plastic smoothing."*

Then dial in. The mask still protects the eyes regardless, but instructing klein to preserve identity reduces drift before the composite even runs.

---

## 5. Models to download (and the license question)

Place under `ComfyUI/models/`:

- **diffusion_models/** `flux-2-klein-4b.safetensors` (distilled, fast)
- **text_encoders/** `qwen_3_4b.safetensors`
- **vae/** `flux2-vae.safetensors`

**License flag for your boss (worth raising explicitly):** FLUX.2 [klein] ships as **4B (Apache 2.0 — commercial use OK)** and **9B (BFL non-commercial license)**. For a commercial product like Magic Hour, the **4B is the safe default**. The 9B is somewhat higher quality but its license likely rules it out for production. Recommend confirming: *"4B Apache for anything we ship?"* I'm confident about the license split per BFL/ComfyUI docs, but licenses change — verify on BFL's model page before relying on it.

---

## 6. Risks & mitigations

- **Likeness drift** — klein regenerates, so without a tight mask the skin pass can subtly shift features. Mitigation: keep the mask composite (already there); keep eyes/brows/lips toggled OFF; keep the feather; consider tightening the skin mask.
- **No denoise knob** — the 0.15–0.35 control disappears. Intensity now comes from prompt wording + mask coverage. This is a real UX change to document for whoever runs the tool.
- **SDXL LoRAs are dead weight** — `better_freckles` and `skin_realism...` are SDXL-only and won't load on Flux. Drop them. Flux2 LoRAs exist but are a separate ecosystem; klein's native skin realism may be enough. (Unverified for *this* use case — test.)
- **New model = newer nodes** — `EmptyFlux2LatentImage`, `Flux2Scheduler`, etc. require an up-to-date ComfyUI. If they're missing, update Comfy in the Colab.
- **Untested graph** — I built this from the real node graphs but could not run it. Treat section 4 as a strong spec, not a guaranteed-working file. Build it in the GUI and load-test.

---

## 7. Suggested test plan

1. **Isolate klein first.** In a fresh graph, load the official *Flux.2 klein 4B image-edit* template, run it on one of your portraits with the skin instruction. Confirm klein loads and runs in your Colab before integrating.
2. **A/B the latent path** — EmptyFlux2LatentImage vs. feeding the encoded crop latent. Pick whichever holds structure better.
3. **Integrate** into the subgraph per section 4.
4. **Compare** using the workflow's built-in `ImageCompare (Before/After)` — check pores improved AND eyes/identity unchanged.
5. **Time it** — confirm klein (4 steps) is actually faster than the old UltimateSDUpscale pass; this was a main goal.

---

## 8. Open questions for your boss

1. **4B (Apache, commercial-safe) or 9B (non-commercial)?** — recommend 4B for production.
2. **Distilled (4-step, fast) or base (20-step, more controllable)?** — distilled fits the "fast" goal; base if quality wins.
3. Is losing the explicit **denoise slider** acceptable, given the mask now governs intensity?
4. Keep **SeedVR2** as the final upscaler, or revisit it too?

---

### Sources
- Reference workflow: `templates_hellorob_facegen_skindetail_upscale.json` (your file)
- ComfyUI Flux.2 Klein 4B Guide — https://docs.comfy.org/tutorials/flux/flux-2-klein
- FLUX.2 [klein] workflow walkthrough — https://comfyui.nomadoor.net/en/basic-workflows/flux-2-klein/
- Official klein 4B image-edit template (node structure reference) — Comfy-Org/workflow_templates
- BFL FLUX.2 [klein] model page — https://bfl.ai/models/flux-2-klein
