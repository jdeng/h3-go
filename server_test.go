package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"image"
	"image/color"
	"image/png"
	"io"
	"math"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func testServer(test *testing.T, runner generationFunc) *videoServer {
	test.Helper()
	directory := test.TempDir()
	root, err := os.OpenRoot(directory)
	if err != nil {
		test.Fatal(err)
	}
	test.Cleanup(func() { root.Close() })
	return &videoServer{settings: options{outputDir: directory}, username: sha256.Sum256([]byte("admin")), password: sha256.Sum256([]byte("test-password")), outputs: root, busy: make(chan struct{}, 1), generate: runner}
}

func formRequest(test *testing.T, fields map[string]string, images map[string][]byte) *http.Request {
	test.Helper()
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	for name, value := range fields {
		if err := writer.WriteField(name, value); err != nil {
			test.Fatal(err)
		}
	}
	for name, data := range images {
		part, err := writer.CreateFormFile(name, "reference.png")
		if err != nil {
			test.Fatal(err)
		}
		_, _ = part.Write(data)
	}
	if err := writer.Close(); err != nil {
		test.Fatal(err)
	}
	request := httptest.NewRequest(http.MethodPost, "/api/generate", &body)
	request.Header.Set("Content-Type", writer.FormDataContentType())
	request.SetBasicAuth("admin", "test-password")
	return request
}

func testPNG(test *testing.T) []byte {
	test.Helper()
	picture := image.NewNRGBA(image.Rect(0, 0, 12, 8))
	picture.Set(0, 0, color.NRGBA{R: 255, A: 128})
	var buffer bytes.Buffer
	if err := png.Encode(&buffer, picture); err != nil {
		test.Fatal(err)
	}
	return buffer.Bytes()
}

func TestSettingsAndFrameGrid(test *testing.T) {
	for _, duration := range []struct {
		seconds string
		frames  int
	}{{"0.2", 5}, {"2", 56}, {"5", 124}, {"10", 243}, {"15", 362}} {
		settings, err := requestSettings(options{}, map[string]string{"prompt": "test", "duration": duration.seconds}, nil)
		if err != nil || settings.frames != duration.frames || settings.width != 640 || settings.height != 384 || settings.cfg != 1 || settings.steps != 8 || !settings.turbo || !settings.audio || !settings.quiet {
			test.Fatalf("defaults/grid: %+v %v", settings, err)
		}
	}
	settings, err := requestSettings(options{}, map[string]string{"prompt": "test", "turbo": "false", "audio": "false", "seed": "9223372036854775807"}, map[string]*referenceImage{"last_frame": {width: 8, height: 12}})
	if err != nil || settings.turbo || settings.audio || settings.steps != 20 || settings.seed != math.MaxInt64 || settings.width != 416 || settings.height != 640 {
		test.Fatalf("standard/portrait/seed: %+v %v", settings, err)
	}
}

func TestInvalidSettings(test *testing.T) {
	for _, override := range []map[string]string{
		{"prompt": ""}, {"prompt": "bad\x00prompt"}, {"prompt": strings.Repeat("中", 8001)},
		{"duration": "NaN"}, {"duration": "16"}, {"steps": "0"}, {"steps": "51"}, {"seed": "-1"}, {"seed": "9223372036854775808"},
		{"cfg": "Inf"}, {"cfg": "0.5"}, {"flow_shift": "0"}, {"lora_strength": "2.1"}, {"turbo": "maybe"}, {"audio": "maybe"},
		{"width": "640"}, {"width": "641", "height": "384"}, {"width": "1536", "height": "1536"}, {"aspect_ratio": "3:1"},
	} {
		values := map[string]string{"prompt": "valid"}
		for name, value := range override {
			values[name] = value
		}
		if _, err := requestSettings(options{}, values, nil); err == nil {
			test.Errorf("accepted invalid settings: %v", override)
		}
	}
}

func TestHTTPKeyframesAndPrivacy(test *testing.T) {
	called := false
	server := testServer(test, func(requestContext context.Context, settings options, first, last *referenceImage) (report, error) {
		called = true
		if first == nil || last == nil || first.width != 12 || len(first.pixels) != 12*8*3 || settings.width != 576 || settings.height != 384 || settings.frames != 243 || !settings.quiet || settings.reportPath != "" {
			test.Fatalf("wrong keyframes/settings: %+v %+v %+v", settings, first, last)
		}
		if err := os.WriteFile(settings.output, []byte("test MP4"), 0600); err != nil {
			return report{}, err
		}
		return report{Frames: 243, FPS: 24, TotalSeconds: 1.5, Output: settings.output}, nil
	})
	request := formRequest(test, map[string]string{"prompt": "private prompt", "width": "576", "height": "384", "seed": "9007199254740993"}, map[string][]byte{"first_frame": testPNG(test), "last_frame": testPNG(test)})
	recorder := httptest.NewRecorder()
	server.handler().ServeHTTP(recorder, request)
	if recorder.Code != http.StatusOK || !called || strings.Contains(recorder.Body.String(), "private prompt") || strings.Contains(recorder.Body.String(), server.settings.outputDir) {
		test.Fatalf("generation/privacy: %d %s", recorder.Code, recorder.Body.String())
	}
	var response map[string]any
	if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
		test.Fatal(err)
	}
	if response["seed_text"] != "9007199254740993" || response["mode"] != "First + last frame" || response["duration"] != 10.125 {
		test.Fatal(response)
	}
	entries, err := os.ReadDir(server.settings.outputDir)
	if err != nil || len(entries) != 1 || !outputName.MatchString(entries[0].Name()) {
		test.Fatalf("unexpected retained files: %v %v", entries, err)
	}
}

func TestHTTPAuthAndCSRF(test *testing.T) {
	server := testServer(test, nil)
	for _, route := range []string{"/", "/outputs/" + strings.Repeat("a", 32) + ".mp4"} {
		recorder := httptest.NewRecorder()
		server.handler().ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, route, nil))
		if recorder.Code != http.StatusUnauthorized || recorder.Header().Get("Cache-Control") != "no-store" {
			test.Fatalf("unprotected %s", route)
		}
	}
	for _, method := range []string{http.MethodPost, http.MethodDelete} {
		request := httptest.NewRequest(method, "/api/generate", nil)
		request.SetBasicAuth("admin", "test-password")
		request.Header.Set("Origin", "https://attacker.example")
		recorder := httptest.NewRecorder()
		server.handler().ServeHTTP(recorder, request)
		if recorder.Code != http.StatusForbidden {
			test.Fatal("cross-origin mutation accepted")
		}
	}
	server.busy <- struct{}{}
	recorder := httptest.NewRecorder()
	server.handler().ServeHTTP(recorder, formRequest(test, map[string]string{"prompt": "test"}, nil))
	if recorder.Code != http.StatusTooManyRequests {
		test.Fatal("parallel generation accepted")
	}
}

func TestHTTPOutputRangeDeleteAndTraversal(test *testing.T) {
	server := testServer(test, nil)
	filename := strings.Repeat("a", 32) + ".mp4"
	if err := os.WriteFile(filepath.Join(server.settings.outputDir, filename), []byte("0123456789"), 0600); err != nil {
		test.Fatal(err)
	}
	request := httptest.NewRequest(http.MethodGet, "/outputs/"+filename, nil)
	request.SetBasicAuth("admin", "test-password")
	request.Header.Set("Range", "bytes=2-5")
	recorder := httptest.NewRecorder()
	server.handler().ServeHTTP(recorder, request)
	if recorder.Code != http.StatusPartialContent || recorder.Body.String() != "2345" || recorder.Header().Get("Content-Type") != "video/mp4" {
		test.Fatalf("video range: %d %s", recorder.Code, recorder.Body.String())
	}
	link := strings.Repeat("b", 32) + ".mp4"
	if err := os.Symlink(filepath.Join(test.TempDir(), "outside.mp4"), filepath.Join(server.settings.outputDir, link)); err != nil {
		test.Fatal(err)
	}
	request = httptest.NewRequest(http.MethodGet, "/outputs/"+link, nil)
	request.SetBasicAuth("admin", "test-password")
	recorder = httptest.NewRecorder()
	server.handler().ServeHTTP(recorder, request)
	if recorder.Code != http.StatusNotFound {
		test.Fatal("symlink escape allowed")
	}
	request = httptest.NewRequest(http.MethodDelete, "/api/outputs/"+filename, nil)
	request.SetBasicAuth("admin", "test-password")
	recorder = httptest.NewRecorder()
	server.handler().ServeHTTP(recorder, request)
	if recorder.Code != http.StatusOK {
		test.Fatal("delete failed")
	}
	if _, err := os.Stat(filepath.Join(server.settings.outputDir, filename)); !os.IsNotExist(err) {
		test.Fatal("video retained after delete")
	}
}

func TestHTTPErrorAndCancellationCleanup(test *testing.T) {
	server := testServer(test, func(requestContext context.Context, settings options, first, last *referenceImage) (report, error) {
		_ = os.WriteFile(settings.output, []byte("partial"), 0600)
		return report{}, errors.New("private prompt and sensitive native path")
	})
	recorder := httptest.NewRecorder()
	server.handler().ServeHTTP(recorder, formRequest(test, map[string]string{"prompt": "private prompt"}, nil))
	if recorder.Code != http.StatusInternalServerError || strings.Contains(recorder.Body.String(), "private prompt") {
		test.Fatal("native error leaked")
	}
	entries, _ := os.ReadDir(server.settings.outputDir)
	if len(entries) != 0 || len(server.busy) != 0 {
		test.Fatal("failed request left output or busy lock")
	}
	server.generate = func(requestContext context.Context, settings options, first, last *referenceImage) (report, error) {
		if requestContext.Err() == nil {
			test.Fatal("request cancellation not propagated")
		}
		return report{}, requestContext.Err()
	}
	request := formRequest(test, map[string]string{"prompt": "test"}, nil)
	requestContext, cancel := context.WithCancel(request.Context())
	cancel()
	server.handler().ServeHTTP(httptest.NewRecorder(), request.WithContext(requestContext))
}

func TestHTTPInvalidUpload(test *testing.T) {
	server := testServer(test, func(context.Context, options, *referenceImage, *referenceImage) (report, error) {
		test.Fatal("invalid upload reached inference")
		return report{}, nil
	})
	for _, images := range []map[string][]byte{{"first_frame": []byte("not PNG")}, {"images": testPNG(test)}, {"first_frame": bytes.Repeat([]byte{0}, maxUploadBytes+1)}} {
		recorder := httptest.NewRecorder()
		server.handler().ServeHTTP(recorder, formRequest(test, map[string]string{"prompt": "test"}, images))
		if recorder.Code != http.StatusBadRequest && recorder.Code != http.StatusRequestEntityTooLarge {
			test.Fatalf("invalid upload accepted: %d", recorder.Code)
		}
		if len(images["first_frame"]) > maxUploadBytes && recorder.Code != http.StatusRequestEntityTooLarge {
			test.Fatal("oversized upload did not return 413")
		}
	}
}

func TestImageDecodeAndWhiteAlpha(test *testing.T) {
	pixels, width, height, err := decodePixels(bytes.NewReader(testPNG(test)), 96)
	if err != nil || width != 12 || height != 8 || len(pixels) != 288 || pixels[0] != 255 || pixels[1] != 127 || pixels[2] != 127 {
		test.Fatalf("decoded image mismatch: %v %d %d %v", pixels[:3], width, height, err)
	}
	if _, _, _, err := decodePixels(bytes.NewReader(testPNG(test)), 95); err == nil {
		test.Fatal("pixel limit ignored")
	}
}

func TestMultipartDuplicates(test *testing.T) {
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	_ = writer.WriteField("prompt", "one")
	_ = writer.WriteField("prompt", "two")
	_ = writer.Close()
	request := httptest.NewRequest(http.MethodPost, "/", &body)
	request.Header.Set("Content-Type", writer.FormDataContentType())
	if _, _, err := readMultipart(request); err == nil {
		test.Fatal("duplicate prompt accepted")
	}
	request = httptest.NewRequest(http.MethodPost, "/", io.NopCloser(strings.NewReader("bad body")))
	if _, _, err := readMultipart(request); err == nil {
		test.Fatal("non-multipart accepted")
	}
}
