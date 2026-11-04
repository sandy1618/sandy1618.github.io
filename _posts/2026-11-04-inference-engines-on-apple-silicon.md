---
layout: post
description: "Four inference engines, one MoE model, one MacBook: websites per hour on a real long-prompt JSON extraction job."
---

# llama.cpp vs BaseRT vs colibri: local LLM inference on an M1 Max, measured on a real workload

If you follow local-LLM news, you have seen the claims.

- **BaseRT** says it beats llama.cpp on Apple Silicon, with faster decode and much faster prefill, plus continuous
  batching built in. <!-- LINK TODO: exact BaseRT claim page; verify wording and which chip the numbers are for -->
- **colibri** says a pure-C engine can run frontier-size MoE models, far bigger than your RAM, on an ordinary machine by
  streaming experts from SSD. <!-- LINK TODO: exact colibri README claim -->

Both claims are probably true for the workload each project measured. But a benchmark only tells you about the job it
ran, and I was not asking "which engine is fastest?" I was asking whether *my* job works on *my* laptop.

My job: I extract structured facts from websites. A model reads a site's key pages and writes a JSON record, each item
with a quote from the page. Cloud models do it well, but I have hundreds of thousands of sites, and a MacBook Pro with
an M1 Max and 64 GB sitting next to the crawler. So: **for this use case, how fast can that laptop do the job, and
which inference engine should run it?**

I tried the claims on this task, locally, with the same model and the same websites on four engines. Here is what I
measured, and where each claim did or did not hold for this use case:

- **llama.cpp (`llama-server`) won**: fastest overall, the only one that gained much from serving 8 requests at once,
  and it never dropped a request or returned broken JSON.
- **BaseRT**: the faster-decode claim held (it writes tokens a little faster than llama.cpp), but it reads prompts at
  half the speed, and on my long prompts that decided it. The batching gain was small here (+12%).
- **colibri**: built for a different problem (models bigger than your RAM). This model fits in RAM, so on this Mac it
  ran ~100x slower. Not a flaw in the claim, just not my use case.
- **Ollama** was the easiest to start and the first to break.

The takeaway is not that those claims are wrong. It depends on your use case, and this post is what happened on mine.

![Sites extracted per hour on the same 20 websites: llama.cpp 181 with 8 requests at once and 124 with one; BaseRT 126 and 113; colibri under 4](/assets/charts/throughput.svg)

## The workload, and why it is not a chat benchmark

Most published numbers are "tokens per second" on a short prompt. My requests look different:

| | Per website |
|---|---|
| Prompt (instructions + page text) | ~4,500 tokens on average, up to ~6,000 |
| Answer | ~1,100-1,400 tokens of JSON |
| Output format | must match a JSON schema (I ask the engine to enforce it) |
| Settings | temperature 0, "thinking" off, 24,576-token context per request |

So every request is **prefill-heavy** (reading a long prompt) *and* decode-heavy (writing a long answer), and the
engine has to support **schema-constrained decoding** (a grammar that only lets the model write valid JSON for my
schema). I measured what matters for a batch job: **websites per hour**, wall clock, start to end.

**Hardware.** MacBook Pro, Apple M1 Max (10 CPU cores, 24-core GPU), 64 GB unified memory.

**Model.** Qwen 3.6 35B-A3B: a mixture-of-experts (MoE) model with 35B parameters, of which ~3B are used per token.
In 4-bit it is ~20 GB, so it fits in memory with room for 8 parallel contexts. Each engine used its own 4-bit file
(GGUF Q4_K_M for llama.cpp and Ollama, BaseRT's `default-q4`, colibri's int4 container), so the four are not
bit-identical models.

**Method.** The same 20 websites through each engine and setting, one engine at a time on an otherwise idle GPU. Each
configuration got one warm-up request first (kernel and grammar compilation), not counted.

## The engines

| Engine | Version | What it is | How I ran it |
|---|---|---|---|
| [llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server` | 0.6.0 (Homebrew) | the C/C++ reference for GGUF models, Metal backend | `-np 1` or `-np 8`, flash attention on, schema via `response_format` |
| [BaseRT](https://basecompute.co) | 0.3.0 | a newer engine for Apple Silicon (closed engine, open CLI), continuous batching | `basert serve --continuous-batching 8`, schema via `response_format` |
| [colibri](https://github.com/JustVugg/colibri) | 2.0.0 | pure C, built to run huge MoE models by streaming experts from disk | `coli serve --ctx 24576 --cap 256` (all experts in RAM) |
| [Ollama](https://ollama.com) | 0.40.2 | the friendly wrapper (GGUF and MLX engines) | `/api/chat` with a JSON schema |

All of them speak an OpenAI-compatible (or Ollama) HTTP API, so one client script drove every run.

## Results

### Throughput

| Engine | Requests at once | Valid JSON | Seconds per site, median | Sites per hour |
|---|---:|---:|---:|---:|
| **llama.cpp** | **8** | 20/20 | 141 (each request waits for the others) | **181** |
| llama.cpp | 1 | 20/20 | 27 | 124 |
| BaseRT | 8 | 20/20 | 183 | 126 |
| BaseRT | 1 | 20/20 | 32 | 113 |
| colibri | 1 | n/a | one site did not finish in 15 minutes | < 4 |
| Ollama (GGUF engine) | 1 (it queues the rest) | ok | ~27 | ~134 (different 50-site sample) |

On a longer run of 200 websites, llama.cpp with 8 slots held **~206 sites/hour** with this model; the 20-site run ends
with a tail where only a few slots are still busy, which pulls its average down.

Two things stand out. **Batching pays on llama.cpp** (124 → 181, +46%) **and barely on BaseRT** (113 → 126, +12%).
And one at a time, llama.cpp is already ahead despite BaseRT's faster decode. The next chart shows why.

### Prefill vs decode

![Prefill: llama.cpp about 570 tokens per second, BaseRT 300, colibri 4.5. Decode: llama.cpp 56, BaseRT 66, colibri about 5](/assets/charts/prefill_decode.svg)

From each engine's own logs, one request at a time:

| Engine | Prefill (reading the prompt) | Decode (writing the answer) | Decode per request, 8 at once |
|---|---:|---:|---:|
| llama.cpp | ~570 tok/s | 56 tok/s | 13 tok/s (~100 tok/s in total) |
| BaseRT | ~300 tok/s | 66 tok/s | 14 tok/s, but prefill drops to ~80 tok/s |
| colibri | ~4.5 tok/s | ~5 tok/s (estimated) | one request at a time only |

With a 4,500-token prompt, llama.cpp spends ~8 s reading and BaseRT ~15 s. BaseRT wins back ~3 s while writing
1,300 tokens. Net: llama.cpp is faster per site. **On long-prompt workloads, look at prefill speed first**; a decode
benchmark on a one-line prompt will pick the wrong engine.

A synthetic probe tells the same story at the extremes: a 1,029-token prompt took **1.8 s** on llama.cpp and
**228 s** on colibri; a 5,029-token prompt took 7.1 s on llama.cpp.

### The model matters as much as the engine

![On llama.cpp with 8 requests at once: Gemma 4 26B-A4B 247 sites per hour, Qwen 3.6 35B-A3B 206, dense Qwen 3.8 27B 37](/assets/charts/models.svg)

Same engine, same 8 slots, 200 websites. Gemma 4 26B-A4B (MoE, ~4B active) ran at **247 sites/hour**, mostly because it
writes ~20% shorter answers. The dense Qwen 3.8 27B, a smaller file than Qwen 3.6 (19 vs 21 GB), ran at **37 sites/hour**: every one of
its 27B parameters works on every token (~40 tok/s prefill, ~2.8 tok/s decode per slot). On a laptop GPU, **active
parameters per token** set the speed far more than total size.

## Engine by engine: what I learned

### llama.cpp: the boring winner

- `llama-server -np 8 -c <8 × context>` serves 8 requests from one copy of the weights. Going from 4 to 8 slots gave a
  small gain only: the GPU, not the slot count, is the limit.
- Schema-constrained JSON via `response_format: {type: json_schema}` worked every time: 440 of 440 answers across my Qwen 3.6
  and Gemma 4 runs parsed and matched the schema.
- It never crashed, hung or leaked memory over multi-hour runs.
- One gotcha: Ollama's Gemma 4 GGUF does not load in llama.cpp ("wrong number of tensors"); the `ggml-org` GGUF on
  Hugging Face does.

### BaseRT: fast decode, slow prefill, and an M1 gap

- Installs as a single CLI with a Metal library; `basert pull` fetches pre-converted models (`basert pull id:variant`
  returned HTTP 401 for me; pulling without the variant worked).
- Its OpenAI-compatible server **rejects any model name other than the one it serves** (HTTP 404, which looks like a
  wrong URL). Send `basecompute/Qwen3.6-35B-A3B`, not your own label.
- `response_format` with a JSON schema is compiled to a grammar: 40/40 valid, and the grammar did not slow decoding.
- It **caches prompt prefixes**: sending the same site twice took 16 s instead of 55 s. Great for chat and agents,
  useless when every prompt is new.
- **Gemma 4 does not run properly on an M1.** Its Metal library has no `moe_simd_gelu_gate_up_q4_sbf16` kernel, so each
  step falls back to a slow path: 24 tok/s prefill, 4.7 tok/s decode, 491 s for one site, where llama.cpp runs the
  same model at ~247 sites/hour. Likely a bf16 kernel aimed at newer Apple chips. I stopped that run.
- With the same model, BaseRT and llama.cpp agreed on the extracted legal name for 17 of 20 sites and the founded year
  for 15 of 20: two different 4-bit quantisations are not quite the same model.

### colibri: the right tool for a different job

colibri runs mixture-of-experts models far larger than RAM (hundreds of billions to trillions of parameters): it keeps
the dense part in memory and streams experts from disk. That is a remarkable trick, and not my situation, since my
model fits in RAM. Still, its setup recommended exactly this model for my machine, so I measured it:

- The one-step setup builds a CPU engine. `make qwen36 METAL=1` builds with Metal, **but the Qwen 3.6 engine uses Metal
  only for its "qpack" weight format**, not for the int4 file the setup downloads. So it ran on the CPU: ~750% CPU,
  12.7 GB resident, ~4.5 tok/s prefill.
- That is in line with colibri's own CPU numbers for this model (6 tok/s decode on a Ryzen 7 desktop): not a broken
  install, just CPU inference against a GPU engine.
- The server runs **one generation at a time**, has **no JSON grammar** for this model family (`response_format`
  returns 400), and, like BaseRT, answers 404 unless you send its `--model-id`.
- Defaults to change for long prompts: context is 8,192 tokens and the expert cache 8 per layer; I used
  `--ctx 24576 --cap 256`.

Where colibri should shine on this Mac is a 284B-744B model streamed from a fast SSD at 0.5-2 tok/s: too slow for a
batch job, interesting for a hard case that no smaller model gets right. I kept it installed for that.

### Ollama: easy to start, first to break

- Its **MLX engine crashed** on an M1 with prompts this long ("Maximum threads per threadgroup is 896 but requested
  1024", then "cache: offset exceeds stored keys").
- Its **GGUF engine** works but serves this model architecture one request at a time ("architecture does not currently
  support parallel requests"): ~27 s per site, ~134 sites/hour, with no way to batch.

### Not yet tested: vLLM

vLLM is the engine I trust most on Linux GPUs; its Apple plugin (vllm-metal, MLX-based) is next on my list as a
reference point.

## Lessons for running LLM batch jobs on Apple Silicon

1. **Benchmark your real prompt length.** Prefill speed differed 2x between the two working engines and decided the
   result; short-prompt "tok/s" charts would have picked the slower one.
2. **Measure sites (or documents) per hour, wall clock.** Per-request latency goes *up* with batching while throughput
   goes up too; only the end-to-end rate tells you what you get.
3. **Check that batching really batches.** llama.cpp gained 46% from 8 slots, BaseRT 12%, colibri and Ollama none.
4. **Pick a mixture-of-experts model.** On the same engine a dense 27B model was 5-7x slower than 26-35B MoE models.
5. **Use schema-constrained decoding** when you need JSON: llama.cpp and BaseRT both enforce a JSON schema; with it, I
   never had to repair an answer.
6. **Expect sharp edges in newer engines:** model-name 404s, missing kernels for older chips, engines that only use the
   GPU for some weight formats. Read the server log before trusting a number.
7. **Run one engine at a time** and warm it up first; two engines sharing the GPU make both numbers meaningless.

## And the quality question

Speed was only half of my question. On a hand-labelled set, the local models' answers were good but not good enough
for data I publish: after my cleaning rules, 87-98% of the legal names, founding years and people they kept were
correct (Qwen 3.6: 92.7%, 91.7%, 95.5%), against 98-100% for the cloud models I use. The errors are the subtle kind:
the parent company's founding year instead of the division's, a guest speaker listed as staff, a brand name taken for a
legal name. So for now the laptop stays a test bench, and llama.cpp is the engine on it.

## Appendix: how to reproduce

- Client: one script sends each website to an OpenAI-compatible `/v1/chat/completions` with the JSON schema in
  `response_format`, temperature 0, `chat_template_kwargs: {enable_thinking: false}`, and records seconds and token
  counts per site.
- Servers, as run:

```bash
llama-server -m qwen3.6-35b-a3b-q4_K_M.gguf -np 8 -c 196608 -fa on --jinja --port 8081
basert serve basecompute/Qwen3.6-35B-A3B --port 8453 --continuous-batching 8 --max-context 24576
coli serve --model qwen36-35b --model-id qwen36 --ctx 24576 --cap 256 --port 8000
```

- Caveats: one machine, one run per configuration, 20 websites for the engine comparison (200 for the model chart), a
  different 4-bit file per engine, and colibri's decode rate estimated from a short request (prompt time subtracted).
  The Qwen 3.8 27B number was measured in a separate run with the same llama.cpp setup.
