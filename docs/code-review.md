# Code review and simplification

Scope: the Go/CGO service, browser UI, build/patch workflow, and the local native
changes relative to the pinned sd.cpp and GGML revisions. This is a source review,
not a claim that every model, backend, resolution, or precision is validated.

## Findings

### Fixed: asynchronous keyframe selection

Image decoding previously committed whichever upload finished last, even if it
was an older selection or had already been removed. Generation could also start
while decoding, and two simultaneous uploads checked stale combined-size limits.
The UI now tracks the current load per slot, disables generation while a current
load is pending, and checks limits against the latest committed opposite slot.
One function updates the selected reference and its preview. The redundant
`DataTransfer` copy into the file input is gone; request files come from the
reference state, as before. Obsolete object URLs are released.

### Fixed: deletion response clearing a newer result

A slow delete request for an old video could finish after a new video appeared
and clear that newer preview. Deletion now retains the original result identity
and only clears the preview if it still displays that result.

### Open: cancellation immediately before native generation starts

`native.go` registers cancellation before entering `h3_run`. Upstream
`src/pipeline/video.cpp` clears the native cancellation flag at the start of
`generate_video`. If cancellation lands between those operations, that reset can
erase it and inference may run to completion before Go reports cancellation.
This race existed before the cleanup; replacing the watcher with
`context.AfterFunc` does not fix it. Cancellation during model creation also
waits for initialization to return. A proper fix needs a latched request-level
cancel signal or a native start/reset handshake, not removal of the lifetime
join or an assumed delay. These cases were identified by source inspection;
the precise start-boundary race was not reproduced on the GPU in this review.

## Simplifications applied

- Native model ownership now has one scoped function and one unconditional
  `defer` for destruction. Cancellation is joined before freeing the engine;
  the native mutex still protects global callbacks but no longer covers FFmpeg.
- CLI validation is performed by `generate` instead of twice. HTTP validation
  is intentionally retained so invalid requests receive a 400 before inference.
- The README separates current setup/results from historical benchmark rounds.
  Earlier measurements remain available in `optimization-history.md`.

These changes simplify ownership and state, not the numerical graph. Production
Go/HTML changes are +73/-67 lines, excluding documentation and regression tests;
this is not a large code-size reduction or a claimed inference speedup.

## Further simplification worth considering

1. The inherited W4A8/W4A4 and Qwen-specific paths are not used by the installed
   H3 INT8 diffusion + Q4_K_M text encoder. A dedicated H3-only patch set could
   remove that maintenance surface. Keep the shared INT8/ConvRot fixes; removing
   the two PR patches wholesale would also remove prerequisites for later patches.
2. The 18 incremental patches could be consolidated by subsystem while retaining
   PR provenance and exact resulting source hashes. This reduces replay and
   upgrade bookkeeping, not the compiled kernel complexity. It is optional;
   preserving reviewable logical boundaries matters more than the patch count.
3. Keep the small standard-library server and CGO bridge. Splitting them into a
   general job framework, codec abstraction, or frontend framework is not useful
   for the current single-GPU, single-job service.

Do not simplify away pinned-transfer events, segment-end synchronization,
allocator alias guards, CUDA shape/type eligibility, unfused fallbacks, or atomic
shared-library replacement. Those guard correctness and running-service safety.

## Validation in this review

- Seven deterministic UI tests pass; six failed against the prior UI.
- `scripts/test.sh` passes Go race tests and `go vet`.
- The standalone Go binary builds against the existing native libraries.
- All native source manifests and patch checksums still match: native numerical
  code is unchanged. No full GPU benchmark or native kernel suite was rerun.
- The live service was left running; the reviewed binary was built separately
  under `build/` and has not been deployed or restarted.
