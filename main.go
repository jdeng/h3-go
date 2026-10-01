package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"unicode/utf8"
)

type options struct {
	prompt, negative, input, endInput, output, models, reportPath string
	listen, outputDir                                             string
	width, height, steps, frames                                  int
	seed                                                          int64
	maxVRAM, duration                                             float64
	cfg, flowShift, loraStrength                                  float64
	turbo, audio, quiet                                           bool
}

type report struct {
	InferenceSeconds float64 `json:"inference_seconds"`
	TotalSeconds     float64 `json:"total_seconds"`
	Frames           int     `json:"frames"`
	FPS              int     `json:"fps"`
	Output           string  `json:"output"`
}

func run() error {
	var config options
	flag.StringVar(&config.prompt, "prompt", "", "Chinese or English prompt")
	flag.StringVar(&config.input, "input-image", "", "Optional first-frame PNG or JPEG")
	flag.StringVar(&config.endInput, "end-image", "", "Optional last-frame PNG or JPEG")
	flag.StringVar(&config.negative, "negative-prompt", "", "Negative prompt; only effective with CFG above 1")
	flag.StringVar(&config.models, "models", "models", "Shared model directory")
	flag.StringVar(&config.output, "output", "outputs/output.mp4", "Output MP4")
	flag.StringVar(&config.reportPath, "report", "", "Optional timing JSON (no prompt)")
	flag.IntVar(&config.width, "width", 640, "Width, multiple of 32")
	flag.IntVar(&config.height, "height", 384, "Height, multiple of 32")
	flag.IntVar(&config.steps, "steps", 8, "Sampling steps; defaults to 20 when Turbo is disabled")
	flag.IntVar(&config.frames, "frames", 0, "Frame count; overrides duration, aligned to 17k+5")
	flag.Float64Var(&config.duration, "duration", 2.333333, "Requested seconds at 24 fps")
	flag.Float64Var(&config.maxVRAM, "max-vram", 0, "Managed GPU budget in GiB; default zero uses live available VRAM")
	flag.Int64Var(&config.seed, "seed", 42, "Seed")
	flag.Float64Var(&config.cfg, "cfg", 1, "Classifier-free guidance; default 1")
	flag.Float64Var(&config.flowShift, "flow-shift", 12, "Video diffusion flow shift")
	flag.Float64Var(&config.loraStrength, "lora-strength", 1, "Turbo LoRA strength")
	flag.BoolVar(&config.turbo, "turbo", true, "Use the installed 8-step Turbo LoRA")
	flag.BoolVar(&config.audio, "audio", true, "Decode and include generated stereo audio")
	flag.StringVar(&config.listen, "serve", "", "Serve HTTP, e.g. 0.0.0.0:8092")
	flag.StringVar(&config.outputDir, "output-dir", "web-outputs", "HTTP video output directory (separate from public review files)")
	flag.Parse()
	if flag.NArg() != 0 {
		return fmt.Errorf("no positional arguments are accepted")
	}
	if config.listen != "" {
		return serve(config)
	}
	if !config.turbo {
		explicitSteps := false
		flag.Visit(func(value *flag.Flag) { explicitSteps = explicitSteps || value.Name == "steps" })
		if !explicitSteps {
			config.steps = 20
		}
	}
	requestContext, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	result, err := generate(requestContext, config, nil, nil)
	if err != nil {
		return err
	}
	encoded, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		return err
	}
	if config.reportPath != "" {
		if err := os.MkdirAll(filepath.Dir(config.reportPath), 0700); err != nil {
			return err
		}
		if err := os.WriteFile(config.reportPath, append(encoded, '\n'), 0600); err != nil {
			return err
		}
	}
	fmt.Println(string(encoded))
	return nil
}

func normalizeOptions(config *options) error {
	for _, prompt := range []string{config.prompt, config.negative} {
		if !utf8.ValidString(prompt) || strings.ContainsRune(prompt, '\x00') || utf8.RuneCountInString(prompt) > 8000 {
			return fmt.Errorf("prompts must contain valid UTF-8, no NUL bytes, and at most 8000 characters each")
		}
	}
	if strings.TrimSpace(config.prompt) == "" {
		return fmt.Errorf("prompt is required")
	}
	if config.width <= 0 || config.height <= 0 || config.width > 8192 || config.height > 8192 || config.width%32 != 0 || config.height%32 != 0 {
		return fmt.Errorf("width and height must be positive multiples of 32, at most 8192")
	}
	if config.steps < 1 || config.steps > 1000 || config.frames < 0 || math.IsNaN(config.maxVRAM) || math.IsInf(config.maxVRAM, 0) || config.maxVRAM < 0 {
		return fmt.Errorf("invalid steps, frame count, or VRAM budget")
	}
	if !finiteBetween(config.cfg, 1, 10) || !finiteBetween(config.flowShift, 1, 20) || !finiteBetween(config.loraStrength, 0, 2) || config.seed < 0 {
		return fmt.Errorf("CFG must be in [1, 10], flow shift in [1, 20], LoRA strength in [0, 2], and seed nonnegative")
	}
	if config.frames == 0 {
		if math.IsNaN(config.duration) || math.IsInf(config.duration, 0) || config.duration <= 0 || config.duration > 15 {
			return fmt.Errorf("duration must be greater than zero and at most 15 seconds")
		}
		config.frames = int(math.Ceil(config.duration * 24))
	}
	if config.frames > 362 {
		return fmt.Errorf("frame count exceeds this CLI's 15-second-scale limit")
	}
	config.frames = max(5, config.frames)
	config.frames += (5 - config.frames%17 + 17) % 17
	if !strings.EqualFold(filepath.Ext(config.output), ".mp4") {
		return fmt.Errorf("output must end in .mp4")
	}
	return nil
}

func finiteBetween(value, minimum, maximum float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0) && value >= minimum && value <= maximum
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
