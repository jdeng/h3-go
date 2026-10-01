package main

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	_ "embed"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"math"
	"math/big"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"syscall"
	"time"
)

//go:embed web/index.html
var indexHTML string

const maxUploadBytes = 64 << 20
const maxUploadPixels = 40 * 1024 * 1024

type generationFunc func(context.Context, options, *referenceImage, *referenceImage) (report, error)

type videoServer struct {
	settings           options
	username, password [32]byte
	outputs            *os.Root
	busy               chan struct{}
	generate           generationFunc
}

var outputName = regexp.MustCompile(`^[0-9a-f]{32}\.mp4$`)
var aspectRatios = map[string]float64{"5:3": 5.0 / 3, "1:1": 1, "3:2": 1.5, "2:3": 2.0 / 3, "16:9": 16.0 / 9, "9:16": 9.0 / 16, "4:3": 4.0 / 3, "3:4": 0.75, "21:9": 21.0 / 9}

func respondJSON(writer http.ResponseWriter, status int, value any) {
	writer.Header().Set("Content-Type", "application/json")
	writer.WriteHeader(status)
	_ = json.NewEncoder(writer).Encode(value)
}

func respondError(writer http.ResponseWriter, status int, message string) {
	respondJSON(writer, status, map[string]string{"error": message})
}

func (server *videoServer) handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /{$}", func(writer http.ResponseWriter, request *http.Request) {
		writer.Header().Set("Content-Type", "text/html; charset=utf-8")
		_, _ = io.WriteString(writer, indexHTML)
	})
	mux.HandleFunc("GET /healthz", func(writer http.ResponseWriter, request *http.Request) {
		respondJSON(writer, http.StatusOK, map[string]any{"ok": true, "backend": "h3-go", "busy": len(server.busy) != 0})
	})
	mux.HandleFunc("POST /api/generate", server.handleGenerate)
	mux.HandleFunc("GET /outputs/{filename}", server.handleOutput)
	mux.HandleFunc("DELETE /api/outputs/{filename}", server.handleDelete)
	return http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		writer.Header().Set("X-Content-Type-Options", "nosniff")
		writer.Header().Set("X-Frame-Options", "DENY")
		writer.Header().Set("Referrer-Policy", "no-referrer")
		writer.Header().Set("Cache-Control", "no-store")
		writer.Header().Set("Content-Security-Policy", "default-src 'self'; img-src 'self' blob:; media-src 'self' blob:; style-src 'unsafe-inline'; script-src 'unsafe-inline'; frame-ancestors 'none'; form-action 'self'")
		if request.URL.Path != "/healthz" {
			username, password, present := request.BasicAuth()
			userHash, passwordHash := sha256.Sum256([]byte(username)), sha256.Sum256([]byte(password))
			matches := subtle.ConstantTimeCompare(userHash[:], server.username[:]) & subtle.ConstantTimeCompare(passwordHash[:], server.password[:])
			if !present || matches != 1 {
				writer.Header().Set("WWW-Authenticate", `Basic realm="H3 Go"`)
				respondError(writer, http.StatusUnauthorized, "Authentication required")
				return
			}
		}
		if request.Method == http.MethodPost || request.Method == http.MethodDelete {
			if origin := request.Header.Get("Origin"); origin != "" {
				parsed, err := url.Parse(origin)
				if err != nil || parsed.Host != request.Host || (parsed.Scheme != "http" && parsed.Scheme != "https") {
					respondError(writer, http.StatusForbidden, "Cross-origin requests are not allowed")
					return
				}
			}
			if request.Header.Get("Sec-Fetch-Site") == "cross-site" {
				respondError(writer, http.StatusForbidden, "Cross-site requests are not allowed")
				return
			}
		}
		mux.ServeHTTP(writer, request)
	})
}

func readMultipart(request *http.Request) (map[string]string, map[string]*referenceImage, error) {
	reader, err := request.MultipartReader()
	if err != nil {
		return nil, nil, errors.New("use multipart/form-data")
	}
	values := make(map[string]string)
	references := make(map[string]*referenceImage)
	fields := map[string]bool{"prompt": true, "negative_prompt": true, "resolution": true, "aspect_ratio": true, "width": true, "height": true, "duration": true, "steps": true, "seed": true, "cfg": true, "flow_shift": true, "lora_strength": true, "turbo": true, "audio": true}
	var pixels int64
	for parts := 0; ; parts++ {
		part, err := reader.NextPart()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return nil, nil, err
		}
		if parts >= 20 {
			return nil, nil, errors.New("too many form fields")
		}
		name := part.FormName()
		if name == "first_frame" || name == "last_frame" {
			if part.FileName() == "" {
				part.Close()
				continue
			}
			if references[name] != nil {
				return nil, nil, errors.New("only one image per keyframe is supported")
			}
			data, err := io.ReadAll(part)
			if err != nil {
				return nil, nil, err
			}
			decoded, width, height, err := decodePixels(bytes.NewReader(data), maxUploadPixels-pixels)
			if err != nil {
				return nil, nil, fmt.Errorf("%s: PNG/JPEG required, at most 40 MP combined", name)
			}
			pixels += int64(width) * int64(height)
			references[name] = &referenceImage{decoded, width, height}
		} else {
			if !fields[name] || part.FileName() != "" {
				return nil, nil, errors.New("unsupported form field")
			}
			if _, duplicate := values[name]; duplicate {
				return nil, nil, errors.New("duplicate form field")
			}
			data, err := io.ReadAll(io.LimitReader(part, 32001))
			if err != nil {
				return nil, nil, err
			}
			if len(data) > 32000 {
				return nil, nil, errors.New("form field is too long")
			}
			values[name] = strings.TrimSpace(string(data))
		}
		part.Close()
	}
	return values, references, nil
}

func integerField(values map[string]string, name string, fallback, minimum, maximum int) (int, error) {
	if values[name] == "" {
		return fallback, nil
	}
	value, err := strconv.Atoi(values[name])
	if err != nil || value < minimum || value > maximum {
		return 0, fmt.Errorf("%s must be an integer between %d and %d", name, minimum, maximum)
	}
	return value, nil
}

func floatField(values map[string]string, name string, fallback, minimum, maximum float64) (float64, error) {
	if values[name] == "" {
		return fallback, nil
	}
	value, err := strconv.ParseFloat(values[name], 64)
	if err != nil || !finiteBetween(value, minimum, maximum) {
		return 0, fmt.Errorf("%s must be a number between %g and %g", name, minimum, maximum)
	}
	return value, nil
}

func boolField(values map[string]string, name string, fallback bool) (bool, error) {
	if values[name] == "" {
		return fallback, nil
	}
	if values[name] != "true" && values[name] != "false" {
		return false, fmt.Errorf("%s must be true or false", name)
	}
	return values[name] == "true", nil
}

func requestSettings(base options, values map[string]string, references map[string]*referenceImage) (options, error) {
	settings := base
	settings.prompt, settings.negative, settings.output = values["prompt"], values["negative_prompt"], "pending.mp4"
	settings.input, settings.endInput, settings.reportPath = "", "", ""
	settings.frames, settings.quiet = 0, true
	var err error
	settings.turbo, err = boolField(values, "turbo", true)
	if err != nil {
		return settings, err
	}
	settings.audio, err = boolField(values, "audio", true)
	if err != nil {
		return settings, err
	}
	defaultSteps := 8
	if !settings.turbo {
		defaultSteps = 20
	}
	settings.steps, err = integerField(values, "steps", defaultSteps, 1, 50)
	if err != nil {
		return settings, err
	}
	for _, field := range []struct {
		name                       string
		target                     *float64
		fallback, minimum, maximum float64
	}{{"duration", &settings.duration, 10, 0.2, 15}, {"cfg", &settings.cfg, 1, 1, 10}, {"flow_shift", &settings.flowShift, 12, 1, 20}, {"lora_strength", &settings.loraStrength, 1, 0, 2}} {
		*field.target, err = floatField(values, field.name, field.fallback, field.minimum, field.maximum)
		if err != nil {
			return settings, err
		}
	}
	if values["seed"] == "" {
		seed, err := rand.Int(rand.Reader, new(big.Int).Lsh(big.NewInt(1), 63))
		if err != nil {
			return settings, errors.New("random seed unavailable")
		}
		settings.seed = seed.Int64()
	} else {
		settings.seed, err = strconv.ParseInt(values["seed"], 10, 64)
		if err != nil || settings.seed < 0 {
			return settings, errors.New("seed must be an integer in [0, 9223372036854775807]")
		}
	}
	if values["width"] != "" || values["height"] != "" {
		if values["width"] == "" || values["height"] == "" {
			return settings, errors.New("provide both width and height")
		}
		settings.width, err = integerField(values, "width", 0, 128, 1536)
		if err != nil {
			return settings, err
		}
		settings.height, err = integerField(values, "height", 0, 128, 1536)
		if err != nil {
			return settings, err
		}
	} else {
		resolution, err := integerField(values, "resolution", 640, 256, 1024)
		if err != nil {
			return settings, err
		}
		ratioName := values["aspect_ratio"]
		if ratioName == "" {
			ratioName = "source"
		}
		ratio := 5.0 / 3
		if ratioName == "source" {
			reference := references["first_frame"]
			if reference == nil {
				reference = references["last_frame"]
			}
			if reference != nil {
				ratio = float64(reference.width) / float64(reference.height)
			}
		} else {
			var found bool
			ratio, found = aspectRatios[ratioName]
			if !found {
				return settings, errors.New("unsupported aspect ratio")
			}
		}
		width, height := float64(resolution), float64(resolution)/ratio
		if ratio < 1 {
			width, height = float64(resolution)*ratio, float64(resolution)
		}
		settings.width, settings.height = int(math.Round(width/32))*32, int(math.Round(height/32))*32
	}
	if settings.width < 128 || settings.height < 128 || settings.width > 1536 || settings.height > 1536 || settings.width*settings.height > 1024*1024 {
		return settings, errors.New("HTTP canvas must be 128–1536 pixels per side and at most 1 megapixel; reduce resolution or use a less extreme ratio")
	}
	err = normalizeOptions(&settings)
	return settings, err
}

func (server *videoServer) handleGenerate(writer http.ResponseWriter, request *http.Request) {
	select {
	case server.busy <- struct{}{}:
		defer func() { <-server.busy }()
	default:
		writer.Header().Set("Retry-After", "10")
		respondError(writer, http.StatusTooManyRequests, "A generation is already running; wait for it to finish")
		return
	}
	request.Body = http.MaxBytesReader(writer, request.Body, maxUploadBytes)
	defer request.Body.Close()
	values, references, err := readMultipart(request)
	if err != nil {
		var oversized *http.MaxBytesError
		if errors.As(err, &oversized) {
			respondError(writer, http.StatusRequestEntityTooLarge, "Total upload limit is 64 MiB")
		} else {
			respondError(writer, http.StatusBadRequest, "Invalid upload: "+err.Error())
		}
		return
	}
	settings, err := requestSettings(server.settings, values, references)
	if err != nil {
		respondError(writer, http.StatusBadRequest, err.Error())
		return
	}
	identifier := make([]byte, 16)
	if _, err := rand.Read(identifier); err != nil {
		respondError(writer, http.StatusInternalServerError, "Unable to allocate job ID")
		return
	}
	jobID := hex.EncodeToString(identifier)
	filename := jobID + ".mp4"
	settings.output = filepath.Join(server.settings.outputDir, filename)
	requestContext, cancel := context.WithTimeout(request.Context(), 30*time.Minute)
	defer cancel()
	result, err := server.generate(requestContext, settings, references["first_frame"], references["last_frame"])
	if err != nil || requestContext.Err() != nil {
		_ = server.outputs.Remove(filename)
		log.Printf("job=%s failed or canceled", jobID)
		respondError(writer, http.StatusInternalServerError, "Generation failed or canceled. Try a smaller canvas or shorter duration.")
		return
	}
	mode := "Text to video"
	if references["first_frame"] != nil && references["last_frame"] != nil {
		mode = "First + last frame"
	} else if references["first_frame"] != nil {
		mode = "First frame to video"
	} else if references["last_frame"] != nil {
		mode = "Last frame to video"
	}
	log.Printf("job=%s size=%dx%d frames=%d elapsed=%.2fs", jobID, settings.width, settings.height, result.Frames, result.TotalSeconds)
	respondJSON(writer, http.StatusOK, map[string]any{
		"job_id": jobID, "filename": filename, "video_url": "/outputs/" + filename,
		"width": settings.width, "height": settings.height, "mode": mode,
		"frames": result.Frames, "fps": result.FPS, "duration": float64(result.Frames) / float64(result.FPS),
		"seed_text": strconv.FormatInt(settings.seed, 10), "steps": settings.steps,
		"turbo": settings.turbo, "audio": settings.audio, "cfg": settings.cfg,
		"inference_seconds": result.InferenceSeconds, "elapsed_seconds": result.TotalSeconds,
	})
}

func (server *videoServer) handleOutput(writer http.ResponseWriter, request *http.Request) {
	filename := request.PathValue("filename")
	if !outputName.MatchString(filename) {
		respondError(writer, http.StatusNotFound, "Video not found")
		return
	}
	file, err := server.outputs.Open(filename)
	if err != nil {
		respondError(writer, http.StatusNotFound, "Video not found")
		return
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil || !info.Mode().IsRegular() {
		respondError(writer, http.StatusNotFound, "Video not found")
		return
	}
	writer.Header().Set("Content-Type", "video/mp4")
	http.ServeContent(writer, request, filename, info.ModTime(), file)
}

func (server *videoServer) handleDelete(writer http.ResponseWriter, request *http.Request) {
	filename := request.PathValue("filename")
	if !outputName.MatchString(filename) || server.outputs.Remove(filename) != nil {
		respondError(writer, http.StatusNotFound, "Video not found")
		return
	}
	respondJSON(writer, http.StatusOK, map[string]bool{"deleted": true})
}

func serve(settings options) error {
	if _, err := modelPaths(settings); err != nil {
		return err
	}
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		return errors.New("ffmpeg is required")
	}
	password := os.Getenv("H3_PASSWORD")
	if password == "" {
		return errors.New("set H3_PASSWORD before starting HTTP mode")
	}
	username := os.Getenv("H3_USERNAME")
	if username == "" {
		username = "admin"
	}
	if err := os.MkdirAll(settings.outputDir, 0700); err != nil {
		return err
	}
	root, err := os.OpenRoot(settings.outputDir)
	if err != nil {
		return err
	}
	defer root.Close()
	app := &videoServer{settings: settings, username: sha256.Sum256([]byte(username)), password: sha256.Sum256([]byte(password)), outputs: root, busy: make(chan struct{}, 1), generate: generate}
	serviceContext, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	server := &http.Server{Addr: settings.listen, Handler: app.handler(), ReadHeaderTimeout: 10 * time.Second, ReadTimeout: 2 * time.Minute, WriteTimeout: 31 * time.Minute, IdleTimeout: 2 * time.Minute, MaxHeaderBytes: 16 << 10, BaseContext: func(net.Listener) context.Context { return serviceContext }}
	done, shutdownDone := make(chan struct{}), make(chan struct{})
	go func() {
		defer close(shutdownDone)
		select {
		case <-serviceContext.Done():
			shutdownContext, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			defer cancel()
			if err := server.Shutdown(shutdownContext); err != nil {
				_ = server.Close()
			}
		case <-done:
		}
	}()
	log.Printf("H3 Go listening on %s; no prompt/upload history; videos in %s", settings.listen, settings.outputDir)
	err = server.ListenAndServe()
	close(done)
	<-shutdownDone
	if errors.Is(err, http.ErrServerClosed) {
		return nil
	}
	return err
}
