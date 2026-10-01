package main

/*
#include <stdlib.h>
#include "native.h"
*/
import "C"

import (
	"context"
	"encoding/binary"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"sync"
	"time"
	"unsafe"
)

var nativeMutex sync.Mutex

type referenceImage struct {
	pixels        []byte
	width, height int
}

func inputPixels(path string) ([]byte, int, int, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, 0, 0, err
	}
	defer file.Close()
	return decodePixels(file, 64*1024*1024)
}

func decodePixels(reader io.ReadSeeker, limit int64) ([]byte, int, int, error) {
	config, format, err := image.DecodeConfig(reader)
	if err != nil || (format != "png" && format != "jpeg") || config.Width <= 0 || config.Height <= 0 || int64(config.Width)*int64(config.Height) > limit {
		return nil, 0, 0, fmt.Errorf("invalid or oversized reference image")
	}
	if _, err := reader.Seek(0, 0); err != nil {
		return nil, 0, 0, err
	}
	decoded, _, err := image.Decode(reader)
	if err != nil {
		return nil, 0, 0, err
	}
	bounds := decoded.Bounds()
	if bounds.Dx() != config.Width || bounds.Dy() != config.Height {
		return nil, 0, 0, fmt.Errorf("reference dimensions changed during decoding")
	}
	pixels := make([]byte, bounds.Dx()*bounds.Dy()*3)
	for row := 0; row < bounds.Dy(); row++ {
		for column := 0; column < bounds.Dx(); column++ {
			red, green, blue, alpha := decoded.At(bounds.Min.X+column, bounds.Min.Y+row).RGBA()
			offset := (row*bounds.Dx() + column) * 3
			pixels[offset] = byte((red + 65535 - alpha) >> 8)
			pixels[offset+1] = byte((green + 65535 - alpha) >> 8)
			pixels[offset+2] = byte((blue + 65535 - alpha) >> 8)
		}
	}
	return pixels, bounds.Dx(), bounds.Dy(), nil
}

func modelPaths(config options) ([]string, error) {
	paths := []string{
		"diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
		"text_encoders/qwen3vl_32b_minimax_h3-Q4_K_M.gguf",
		"vae/minimax_h3_video_vae_fp16.safetensors",
		"vae/minimax_h3_audio_vae_fp32.safetensors",
		"loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
	}
	for index, relative := range paths {
		paths[index] = filepath.Join(config.models, relative)
		if (index == 3 && !config.audio) || (index == 4 && !config.turbo) {
			continue
		}
		info, err := os.Stat(paths[index])
		if err != nil || !info.Mode().IsRegular() || info.Size() == 0 {
			return nil, fmt.Errorf("missing model: %s", paths[index])
		}
	}
	return paths, nil
}

func generate(requestContext context.Context, config options, first, last *referenceImage) (report, error) {
	if err := normalizeOptions(&config); err != nil {
		return report{}, err
	}
	started := time.Now()
	var native C.h3_options
	var strings []*C.char
	allocate := func(value string) *C.char {
		pointer := C.CString(value)
		strings = append(strings, pointer)
		return pointer
	}
	defer func() {
		for _, pointer := range strings {
			C.free(unsafe.Pointer(pointer))
		}
	}()
	paths, err := modelPaths(config)
	if err != nil {
		return report{}, err
	}
	for index, target := range []**C.char{&native.diffusion, &native.encoder, &native.video_vae, &native.audio_vae, &native.lora} {
		*target = allocate(paths[index])
	}
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		return report{}, fmt.Errorf("ffmpeg is required: %w", err)
	}
	native.prompt = allocate(config.prompt)
	native.negative = allocate(config.negative)
	native.max_vram = allocate(strconv.FormatFloat(config.maxVRAM, 'f', -1, 64))
	native.width, native.height = C.int(config.width), C.int(config.height)
	native.frames, native.steps = C.int(config.frames), C.int(config.steps)
	native.seed = C.int64_t(config.seed)
	native.cfg, native.flow_shift, native.lora_strength = C.float(config.cfg), C.float(config.flowShift), C.float(config.loraStrength)
	native.turbo, native.audio, native.quiet = C.bool(config.turbo), C.bool(config.audio), C.bool(config.quiet)
	for _, input := range []struct {
		path      string
		reference *referenceImage
		target    *C.sd_image_t
	}{{config.input, first, &native.first_frame}, {config.endInput, last, &native.last_frame}} {
		reference := input.reference
		if reference == nil && input.path != "" {
			pixels, width, height, err := inputPixels(input.path)
			if err != nil {
				return report{}, err
			}
			reference = &referenceImage{pixels, width, height}
		}
		if reference != nil {
			if reference.width <= 0 || reference.height <= 0 || int64(reference.width)*int64(reference.height) > 64*1024*1024 || len(reference.pixels) != reference.width*reference.height*3 {
				return report{}, fmt.Errorf("invalid reference buffer")
			}
			buffer := C.CBytes(reference.pixels)
			defer C.free(buffer)
			*input.target = C.sd_image_t{width: C.uint32_t(reference.width), height: C.uint32_t(reference.height), channel: 3, data: (*C.uint8_t)(buffer)}
		}
	}
	var result C.h3_result
	defer C.h3_free_result(&result)
	if err := sampleVideo(requestContext, &native, &result); err != nil {
		return report{}, err
	}
	inference := time.Since(started).Seconds()
	if err := encodeVideo(requestContext, config.output, &result, config.quiet); err != nil {
		return report{}, err
	}
	return report{InferenceSeconds: inference, TotalSeconds: time.Since(started).Seconds(), Frames: int(result.count), FPS: int(result.fps), Output: config.output}, nil
}

func sampleVideo(requestContext context.Context, native *C.h3_options, result *C.h3_result) error {
	nativeMutex.Lock()
	defer nativeMutex.Unlock()
	if err := requestContext.Err(); err != nil {
		return err
	}
	engine := C.h3_create(native)
	if engine == nil {
		return fmt.Errorf("H3 model initialization failed")
	}
	defer C.free_sd_ctx(engine)
	if err := requestContext.Err(); err != nil {
		return err
	}
	canceled := make(chan struct{})
	stop := context.AfterFunc(requestContext, func() {
		C.sd_cancel_generation(engine, C.SD_CANCEL_ALL)
		close(canceled)
	})
	defer func() {
		if !stop() {
			<-canceled
		}
	}()
	success := bool(C.h3_run(engine, native, result))
	if err := requestContext.Err(); err != nil {
		return err
	}
	if !success {
		return fmt.Errorf("H3 generation failed; see native log")
	}
	return nil
}

func encodeVideo(requestContext context.Context, output string, result *C.h3_result, quiet bool) error {
	if result.count <= 0 || result.frames == nil || result.fps <= 0 {
		return fmt.Errorf("native result has no video")
	}
	if err := os.MkdirAll(filepath.Dir(output), 0700); err != nil {
		return err
	}
	temporary, err := os.MkdirTemp(filepath.Dir(output), ".h3-encode-")
	if err != nil {
		return err
	}
	defer os.RemoveAll(temporary)
	frames := unsafe.Slice(result.frames, int(result.count))
	first := frames[0]
	arguments := []string{"-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", fmt.Sprintf("%dx%d", first.width, first.height), "-framerate", strconv.Itoa(int(result.fps)), "-i", "pipe:0"}
	hasAudio := result.audio != nil && result.audio.data != nil && result.audio.sample_count > 0
	if hasAudio {
		track := result.audio
		samples := unsafe.Slice(track.data, int(track.sample_count)*int(track.channels))
		encoded := make([]byte, len(samples)*4)
		for index, sample := range samples {
			binary.LittleEndian.PutUint32(encoded[index*4:], math.Float32bits(float32(sample)))
		}
		audioPath := filepath.Join(temporary, "audio.f32")
		if err := os.WriteFile(audioPath, encoded, 0600); err != nil {
			return err
		}
		arguments = append(arguments, "-f", "f32le", "-ar", strconv.Itoa(int(track.sample_rate)), "-ac", strconv.Itoa(int(track.channels)), "-i", audioPath)
	}
	encodedPath := filepath.Join(temporary, "video.mp4")
	arguments = append(arguments, "-map", "0:v:0")
	if hasAudio {
		arguments = append(arguments, "-map", "1:a:0", "-c:a", "aac", "-b:a", "128k")
	}
	arguments = append(arguments, "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-map_metadata", "-1", encodedPath)
	command := exec.CommandContext(requestContext, "ffmpeg", arguments...)
	if !quiet {
		command.Stderr = os.Stderr
	}
	input, err := command.StdinPipe()
	if err != nil {
		return err
	}
	if err := command.Start(); err != nil {
		input.Close()
		return err
	}
	var writeError error
	for _, frame := range frames {
		if frame.channel != 3 || frame.width != first.width || frame.height != first.height || frame.data == nil {
			writeError = fmt.Errorf("inconsistent native video frame")
			break
		}
		pixels := unsafe.Slice((*byte)(unsafe.Pointer(frame.data)), int(frame.width)*int(frame.height)*3)
		if _, err := input.Write(pixels); err != nil {
			writeError = err
			break
		}
	}
	input.Close()
	waitError := command.Wait()
	runtime.KeepAlive(result)
	if writeError != nil {
		return writeError
	}
	if waitError != nil {
		return fmt.Errorf("ffmpeg: %w", waitError)
	}
	if err := os.Chmod(encodedPath, 0600); err != nil {
		return err
	}
	if err := requestContext.Err(); err != nil {
		return err
	}
	return os.Rename(encodedPath, output)
}
