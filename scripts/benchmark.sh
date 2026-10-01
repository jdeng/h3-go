#!/usr/bin/env bash
set -euo pipefail
umask 077
ulimit -c 0
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS="${H3_MODELS:-$ROOT/../ComfyUI/models}"
NAME="${H3_BENCH_NAME:-h3-optimized}"
[[ "$NAME" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo "Invalid benchmark name" >&2; exit 1; }
mkdir -p "$ROOT/outputs" "$ROOT/build/bench"
PROMPT='视频描述：晨光照进简洁的厨房，一只红色陶瓷茶壶放在浅色木桌上，壶嘴冒出细细的白色蒸汽。镜头从茶壶正面缓慢向右平移，茶壶和桌面保持稳定，阳光和阴影自然，写实产品摄影，画面清晰，没有人物。声音描述：安静的室内环境，轻微的沸水声，没有人声，没有音乐。 <lora:minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16:1>'
/usr/bin/time -v "$ROOT/dist/bin/sd-cli" -M vid_gen \
    --diffusion-model "$MODELS/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors" \
    --llm "$MODELS/text_encoders/qwen3vl_32b_minimax_h3-Q4_K_M.gguf" \
    --vae "$MODELS/vae/minimax_h3_video_vae_fp16.safetensors" \
    --audio-vae "$MODELS/vae/minimax_h3_audio_vae_fp32.safetensors" \
    --lora-model-dir "$MODELS/loras" --prompt "$PROMPT" \
    --width 640 --height 384 --video-frames 56 --fps 24 \
    --steps 8 --cfg-scale 1 --flow-shift 12 \
    --sampling-method res_multistep --scheduler simple --seed 42 --rng cpu \
    --backend cuda0 --params-backend cpu --max-vram "${H3_MAX_VRAM:-0}" --mmap \
    --fa --diffusion-fa --conditioning-cache-size 0 \
    --vae-tiling --vae-tile-size 256x256 --vae-tile-overlap 0.25 --temporal-tiling \
    --disable-image-metadata --log-level debug \
    --output "$ROOT/build/bench/$NAME.avi" \
    2>&1 | tee "$ROOT/build/bench/$NAME.log"
ffmpeg -hide_banner -loglevel error -y -i "$ROOT/build/bench/$NAME.avi" \
    -map 0:v:0 -map '0:a?' -c:v libx264 -preset fast -crf 18 -pix_fmt yuv420p \
    -c:a aac -b:a 128k -movflags +faststart -map_metadata -1 \
    "$ROOT/outputs/$NAME.mp4"
ffprobe -v error -show_entries stream=codec_name,width,height,r_frame_rate,duration,nb_frames,sample_rate,channels \
    -of json "$ROOT/outputs/$NAME.mp4" > "$ROOT/build/bench/$NAME-probe.json"
