# Aria Voice Runtime Baseline

This note records the current stable speech setup for Project Aria.

## Stable Baseline

Aria now uses Kokoro ONNX as the live conversation voice:

- Provider: `kokoro-onnx`
- ONNX execution provider: `CUDAExecutionProvider`
- Model: `assets/voices/kokoro/kokoro-v1.0.onnx`
- Voices: `assets/voices/kokoro/voices-v1.0.bin`
- Fallback: disabled by default
- Failure mode: fail-loud in `logs/aria.log`

The full Kokoro model is required for the RTX 4070 path. The int8 model is a
CPU-friendly asset, but it performed poorly on CUDA and should not be used for
live conversation on this machine.

## Why This Changed

The original voice path masked runtime issues by falling back to Piper. That
kept Aria audible, but it hid whether Kokoro was actually working. During
testing, this caused confusing behavior: startup voice could work, conversation
could feel delayed, and logs were not always clear about which provider was
responsible.

The new design makes the active provider explicit:

- `TTS_CONVERSATION_PROVIDER = "kokoro"`
- `TTS_FALLBACK_PROVIDER = ""`
- `TTS_FAIL_LOUD = True`
- `KOKORO_DISABLE_PROVIDER_FALLBACK = True`

If Kokoro fails, the log should show the real error instead of silently playing
a Piper response.

## GPU Findings

The RTX 4070 was healthy, but ONNX Runtime was not initially using it.

Observed path:

1. `onnxruntime-directml` exposed `DmlExecutionProvider`, but Kokoro failed on
   `ConvTranspose` nodes.
2. `onnxruntime-gpu[cuda,cudnn]` exposed `CUDAExecutionProvider`.
3. The int8 Kokoro model was still slow and noisy on CUDA.
4. The full `kokoro-v1.0.onnx` model produced fast warm synthesis on CUDA.

Final benchmark after warmup:

```text
provider=kokoro-onnx
63 chars: about 0.5s synthesis
90 chars: about 0.5s synthesis
```

The first synthesis after loading can still take several seconds. That is
acceptable during startup, but repeated conversation replies should be fast.

## Required Packages

Install dependencies with:

```powershell
python -m pip install -r requirements.txt
```

The key runtime package is:

```text
onnxruntime-gpu[cuda,cudnn]>=1.24.4
```

Only one ONNX Runtime package should be installed. Do not keep
`onnxruntime-directml` and `onnxruntime` installed beside `onnxruntime-gpu`
unless you are deliberately testing provider conflicts.

Check providers:

```powershell
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

Expected:

```text
['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
```

## Benchmark

Run:

```powershell
python tools\benchmark_tts.py --provider kokoro
```

Healthy output should include:

```text
provider_chain=['kokoro-onnx']
ONNX Runtime providers available for Kokoro: [...]
provider=kokoro-onnx ...
```

Warm short replies should synthesize in roughly `0.5s` on the RTX 4070. If warm
short replies are taking many seconds, verify the model path and ONNX provider.

## Manual Conversation Smoke Test

Run Aria:

```powershell
python main.py
```

In a second terminal:

```powershell
Get-Content logs\aria.log -Wait -Tail 40
```

Test:

- "analysis mode on"
- "what time is it"
- "conversation mode off"
- "conversation mode on"
- "hello Aria"

Success criteria:

- Aria hears speech.
- Transcription completes.
- Replies log `provider=kokoro-onnx`.
- Kokoro logs `CUDAExecutionProvider`.
- No Piper fallback appears.
- Proactive analyst does not interrupt conversation.

## Proactive Analyst Speech

Spoken proactive insights are disabled for now:

```python
PROACTIVE_ANALYST_SPEAK_INSIGHTS = False
```

This keeps the analyst from competing with live conversation. Stage 3b should
turn analyst output into queued notifications rather than direct speech.

## Troubleshooting

If DirectML is visible but Kokoro fails on `ConvTranspose`, do not use DirectML
for this model.

If CUDA is missing, reinstall:

```powershell
python -m pip uninstall -y onnxruntime onnxruntime-directml onnxruntime-gpu
python -m pip install "onnxruntime-gpu[cuda,cudnn]==1.24.4"
```

If the int8 model is configured and conversation is slow, switch back to:

```python
KOKORO_ONNX_MODEL_PATH = "assets/voices/kokoro/kokoro-v1.0.onnx"
```

If Piper appears in `aria.log`, check:

```python
TTS_CONVERSATION_PROVIDER = "kokoro"
TTS_FALLBACK_PROVIDER = ""
TTS_FAIL_LOUD = True
```
