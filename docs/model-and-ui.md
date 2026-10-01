# MiniMax H3: local capabilities and controls

## Installed pipeline

This service uses **FL2VA**, not Ref2VA. Its diffusion checkpoint jointly models
video and audio. The H3-specific Qwen3-VL encoder turns the prompt into conditions;
it is not a chatbot, prompt-rewriting service, or replacement video generator.
The video and audio VAEs decode the two latent streams into pixels and samples.

| Component | Local checkpoint / format | Purpose |
| --- | --- | --- |
| Diffusion | `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | Pruned FL2VA with INT8 ConvRot linear weights |
| Text encoder | `qwen3vl_32b_minimax_h3-Q4_K_M.gguf` | H3-specific Qwen3-VL-32B family encoder; different quantization from ComfyUI NVFP4/AWQ |
| Video VAE | `minimax_h3_video_vae_fp16.safetensors` | Video latent encoding for keyframes and decoding for output |
| Audio VAE | `minimax_h3_audio_vae_fp32.safetensors` | Stereo waveform decoding; optional when audio output is off |
| Turbo adapter | `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors` | Optional runtime BF16 LoRA, normally strength 1 and 8 steps |

The local loader reports 50 diffusion blocks, hidden width 5376, 56 attention heads
of width 128, FFN width 14336, and a 1025-entry AdaLN curve grid. These describe the
loaded pruned variant; a model-family marketing parameter count is not an exact
count of this checkpoint. The video decoder is a 36-layer transformer, not mainly
a convolutional decoder. These details come from the pinned native implementation
and local loading diagnostics, not a new scan of model weights.

## Available modes

| Inputs in this UI | Mode |
| --- | --- |
| Prompt only | Text-to-video with optional decoded audio |
| First frame + prompt | Animate from a starting keyframe |
| Last frame + prompt | Generate toward an ending keyframe |
| First and last frames + prompt | Generate a transition between keyframes |

First/last images are conditioning targets, not a promise of pixel-identical
decoded endpoints. The native image conversion resizes them to the canvas. Use
**Match keyframe** to minimize aspect-ratio distortion, or prepare matching images.
When both are present, the first frame supplies the automatic aspect ratio.

The official system also has a separate Ref2VA model for multimodal references
and a separate 2K regeneration stage. Installing FL2VA does not enable those
capabilities. This UI does not accept a subject-reference gallery, input videos,
reference audio, arbitrary timeline anchors, or edit masks. It does not download
additional weights or silently switch models. See the
[official H3 overview](https://github.com/MiniMax-AI/MiniMax-H3) and
[Comfy-Org model repository](https://huggingface.co/Comfy-Org/MiniMax-H3).

## Generation parameters

| UI control | Default | Behavior |
| --- | --- | --- |
| Prompt | Empty | Required; up to 8000 UTF-8 characters, sent without automatic rewriting |
| First/last keyframe | None | One PNG/JPEG per slot; upload, paste or drag/drop; remove independently |
| Aspect ratio | Match keyframe; otherwise 5:3 | Rounded to a 32-pixel grid; actual dimensions are displayed before generation |
| Resolution | Longest edge 640 | Presets up to 1024; custom sides 128–1536, canvas at most 1,048,576 pixels |
| Duration | 10 seconds | UI 1–15 seconds; API 0.2–15 for smoke tests; aligned upward to `17k+5` frames |
| FPS | Fixed 24 | Native H3 rate; 5/10/15 requested seconds become 124/243/362 frames |
| Turbo | On | Loads the installed 8-step adapter; switching off removes the adapter |
| Steps | 8 Turbo / 20 Standard | UI/API accept 1–50; low-step Standard runs are tests, not quality recommendations |
| CFG | 1 | Range 1–10; above 1 introduces negative/unconditional guidance and additional work |
| Flow shift | 12 | Range 1–20; changes the video noise schedule, not motion speed or clip duration |
| LoRA strength | 1 | Range 0–2, only used in Turbo mode; not an image-adherence slider |
| Negative prompt | Empty | Active only when CFG is above 1; maximum 8000 characters |
| Seed | Random | Nonnegative signed 64-bit integer; returned as text to avoid JavaScript precision loss |
| Include audio | On | Decode 32-kHz stereo and encode AAC; off skips audio VAE/muxing, not joint diffusion |

For an exact 3:2 canvas use custom 576×384 or 768×512; independently rounding a
preset's dimensions can slightly alter its nominal ratio. A 10-second request is
10.125 seconds before any external trimming. Sub-4-second clips are development
previews; the official target duration is 4–15 seconds. The official guide explains
the frame grid and Turbo speed/quality tradeoff:
[Comfy native H3 guide](https://docs.comfy.org/tutorials/video/minimax/minimax-h3-native).

Keep CFG 1, shift 12, 8 Turbo steps, and 640×384 initially. On this machine the
matched 243-frame request took approximately 52 seconds after sampling-dataflow
optimization. This is a measured example, not an ETA or a claim of parity with
ComfyUI. Other GPU users, larger canvases, more steps, and longer clips affect both
latency and memory. Automatic VRAM management cannot guarantee every allowed
canvas/duration combination will fit.

## HTTP and data lifetime

- The Go binary embeds the English UI and calls the same cgo/native pipeline as
  the CLI. It does not call ComfyUI or a Python inference runtime.
- One request runs at a time; concurrent generation returns HTTP 429. There is
  no persistent queue or prompt/history database. The page shows elapsed time,
  not a fabricated completion percentage.
- A request has a 30-minute timeout. Cancel aborts its HTTP request and asks the
  native engine to stop at a safe boundary. Model loading and an in-flight kernel
  are not instantly interruptible. Generation weights/workspaces are released
  before freeing the slot; the live CUDA runtime retains context overhead.
- Upload bytes and decoded keyframes exist in memory only. Native prompt logs are
  suppressed in HTTP mode. Service logs contain random job IDs, dimensions,
  frame counts, timing, and generic failures, not prompts or filenames supplied
  by the browser.
- Only completed MP4s remain in `web-outputs/`. Encoding scratch directories are
  removed on normal completion/error; SIGKILL or a crash can leave them behind.
  There is no browser localStorage or result listing. Keep a download/link if you
  need it after reloading the page. Delete removes the selected generated MP4.
- Basic authentication protects the UI, generation, download and delete routes.
  `/healthz` exposes only liveness and busy state. MP4 byte ranges support seeking.
  Upload requests are capped at 64 MiB including multipart overhead and 40 MP
  combined decoded images. Output access is restricted to generated filenames
  under an `os.Root`, with no directory listing or filesystem escape.
- `web-outputs/` is deliberately separate from the unauthenticated 8090 review
  server's `outputs/` root. Plain HTTP does not encrypt passwords or videos; use
  a trusted LAN or add TLS before exposing it to an untrusted network.

## API example

```bash
curl -u "admin:$H3_PASSWORD" http://127.0.0.1:8092/api/generate \
  -F 'prompt=视频描述：晨光中的红色茶壶，镜头缓慢推进。声音描述：轻微沸水声。' \
  -F first_frame=@reference.png \
  -F aspect_ratio=source -F resolution=640 -F duration=10 \
  -F turbo=true -F steps=8 -F cfg=1 -F audio=true -F seed=42
```

Optional `last_frame=@ending.png` enables an ending keyframe. Omit both image
fields for T2V. Use `width` and `height` instead of ratio/resolution for an exact
canvas. The synchronous response returns `video_url`, filename, seed text,
actual frame count/duration and timing. It never echoes the prompt.

Use authenticated `GET /outputs/<filename>.mp4` to stream/download and
`DELETE /api/outputs/<filename>.mp4` to delete. Unknown/duplicate form fields and
unsupported image data are rejected before inference. There is no server-side
URL fetching or arbitrary client-chosen output path.
