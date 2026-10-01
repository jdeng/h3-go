#ifndef H3_GO_NATIVE_H
#define H3_GO_NATIVE_H

#include "stable-diffusion.h"

typedef struct {
    const char *diffusion;
    const char *encoder;
    const char *video_vae;
    const char *audio_vae;
    const char *lora;
    const char *prompt;
    const char *negative;
    const char *max_vram;
    int width;
    int height;
    int frames;
    int steps;
    int64_t seed;
    float cfg;
    float flow_shift;
    float lora_strength;
    bool turbo;
    bool audio;
    bool quiet;
    sd_image_t first_frame;
    sd_image_t last_frame;
} h3_options;

typedef struct {
    sd_image_t *frames;
    int count;
    int fps;
    sd_audio_t *audio;
} h3_result;

sd_ctx_t *h3_create(const h3_options *options);
bool h3_run(sd_ctx_t *engine, const h3_options *options, h3_result *result);
void h3_free_result(h3_result *result);

#endif
