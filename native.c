#include "native.h"
#include <stdio.h>
#include <string.h>

static bool quiet_logging;

static void native_log(enum sd_log_level_t level, const char *text, void *data) {
    (void)data;
    if (!quiet_logging && level >= SD_LOG_INFO) {
        fputs(text, stderr);
    }
}

static void native_progress(int step, int steps, float seconds, void *data) {
    (void)step;
    (void)steps;
    (void)seconds;
    (void)data;
}

sd_ctx_t *h3_create(const h3_options *options) {
    quiet_logging = options->quiet;
    sd_set_log_callback(native_log, NULL);
    sd_set_progress_callback(native_progress, NULL);
    sd_ctx_params_t context;
    sd_ctx_params_init(&context);
    context.diffusion_model_path = options->diffusion;
    context.llm_path = options->encoder;
    context.vae_path = options->video_vae;
    context.audio_vae_path = options->audio ? options->audio_vae : NULL;
    context.backend = "cuda0";
    context.params_backend = "cpu";
    context.auto_fit = false;
    context.max_vram = options->max_vram;
    context.enable_mmap = true;
    context.flash_attn = true;
    context.diffusion_flash_attn = true;
    context.rng_type = CPU_RNG;
    context.sampler_rng_type = CPU_RNG;
    context.conditioning_cache_size = 0;
    context.lora_apply_mode = LORA_APPLY_AT_RUNTIME;
    return new_sd_ctx(&context);
}

bool h3_run(sd_ctx_t *engine, const h3_options *options, h3_result *result) {
    memset(result, 0, sizeof(*result));
    sd_vid_gen_params_t generation;
    sd_vid_gen_params_init(&generation);
    sd_lora_t lora = {false, options->lora_strength, options->lora};
    generation.loras = options->turbo ? &lora : NULL;
    generation.lora_count = options->turbo ? 1 : 0;
    generation.prompt = options->prompt;
    generation.negative_prompt = options->negative;
    generation.width = options->width;
    generation.height = options->height;
    generation.video_frames = options->frames;
    generation.fps = 24;
    generation.seed = options->seed;
    generation.init_image = options->first_frame;
    generation.end_image = options->last_frame;
    generation.sample_params.sample_method = RES_MULTISTEP_SAMPLE_METHOD;
    generation.sample_params.scheduler = SIMPLE_SCHEDULER;
    generation.sample_params.sample_steps = options->steps;
    generation.sample_params.flow_shift = options->flow_shift;
    generation.sample_params.guidance.txt_cfg = options->cfg;
    generation.vae_tiling_params.enabled = true;
    generation.vae_tiling_params.temporal_tiling = true;
    generation.vae_tiling_params.tile_size_w = 256;
    generation.vae_tiling_params.tile_size_h = 256;
    generation.vae_tiling_params.target_overlap = 0.25f;
    return generate_video(engine, &generation, &result->frames,
                          &result->count, &result->audio, &result->fps);
}

void h3_free_result(h3_result *result) {
    free_sd_images(result->frames, result->count);
    free_sd_audio(result->audio);
    memset(result, 0, sizeof(*result));
}
