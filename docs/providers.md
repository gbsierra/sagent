# Providers

Sagent separates providers from models. A provider owns authentication and creates model objects. A model sends typed requests and returns typed responses.

## Public provider matrix

| Provider class | Environment variable | Default model | Utility model | Notes |
| --- | --- | --- | --- | --- |
| `Anthropic` | `ANTHROPIC_API_KEY` | `claude-opus-5-5` | `claude-sonnet-5-5` | Anthropic API-key provider; `best` resolves to `claude-fable-5-1`. |
| `OpenAI` | `OPENAI_API_KEY` | `gpt-6.1-sol` | `gpt-6-luna` | OpenAI API provider. |
| `Google` | `GOOGLE_API_KEY` | `gemini-3.1-pro-preview` | `gemini-3.5-flash-lite` | Google Gemini provider. |
| `Moonshot` | `MOONSHOT_API_KEY` | `kimi-k3` | `kimi-k2.6` | OpenAI-compatible Kimi provider. |
| `DashScope` | `DASHSCOPE_API_KEY` | `qwen3.7-plus` | `qwen3.8-flash` | Alibaba DashScope provider. |
| `MiniMax` | `MINIMAX_API_KEY` | `MiniMax-M3` | `MiniMax-M3` | MiniMax provider. |
| `SelfHosted` | none | `Qwen/Qwen3.6-27B` | configured snapshot | Local HF transformers provider. |
| `OpenAICompat` | subclass-defined | subclass-defined | subclass-defined | Base class for chat-completions-compatible APIs. |

The public package is designed around API-key providers.

## Basic usage

```python
from sagent.providers import Google

provider = Google.from_env()
model = provider.model("gemini-3.1-pro-preview")
utility = provider.model("utility")
```

All public API-key providers support:

```python
ProviderClass.from_env()
ProviderClass.from_key("...")
ProviderClass.from_env().model("model-id")
ProviderClass.from_env().model("utility")
```

`model(None)` uses the provider's default model. Unknown model IDs raise with the provider's known models and the concrete row each role names.

Catalog keys are the provider's own model IDs, exactly as its API takes and
reports them: `gpt-6.1-sol`, `claude-opus-5-5`, `gemini-3.1-pro-preview`,
`MiniMax-M3`. Sagent infers nothing from how an ID is spelled, so a family word
such as `opus` is not a model. (`sol` and `luna` work because OpenAI declares
them as explicit aliases, not because Sagent guesses.) The roles `default`, `utility`, and
`best` each name one ID explicitly. When a provider publishes both a moving
alias and the dated snapshot it points at, both are rows with the same facts.
The names Sagent used before (`opus-5.5`, `sol-6.1`) still load from saved
sessions and settings and resolve to the provider ID; sessions save the
provider ID from then on. A swap announcement (`/model`, `AgentSelf`) shows
the resolved ID. A bare model ID selects its largest context window. Append `+272k` for a smaller
OpenAI window or `+200k` for a smaller Anthropic window when the model offers
one. An explicit `+1m` is also accepted when the bare model already provides
that window. Both OpenAI providers use the Responses API. Supported reasoning
efforts vary by model. When a saved effort moves to a model that does not
offer it -- a `/model` swap, a spawned agent, or a resumed session --
Sagent picks the closest effort the new model does offer that still does
some reasoning: `max` on a model whose highest effort is `high` becomes
`high`, not `none`. When two are equally close, it picks the lower one.
Only when the new model offers no reasoning efforts at all does the
selection fall back to `none`.

API-key requests send output-token limits and temperature for non-reasoning
models. The subscription backend omits those unsupported fields. Both paths
replay local conversation history with `store: false`, preserving encrypted
reasoning and image-bearing tool results. Existing sessions need no conversion.

## CLI dispatch

```bash
sagent --provider Google --auth env --model gemini-3.1-pro-preview
```

`--provider` is the provider class name from `sagent.providers` (case-insensitive). `--auth env` calls `Google.from_env()`.

Short aliases work everywhere a provider name is accepted (`--provider`, `--allow-providers`, `/model provider=...`, agent-tool arguments):

| Alias | Provider |
| --- | --- |
| `oai` | `OpenAI` |
| `oaisub` | `OpenAISubscription` |
| `ant` | `Anthropic` |
| `antsub` | `AnthropicCLI` |

Aliases resolve to the canonical class name, which is what sessions persist and the UI displays.

If the named factory does not exist, Sagent reports an error. Use `--auth env`
for API keys so secrets do not land in shell history.

## Provider inference

Agent tools infer a provider from catalog membership, so IDs such as
`claude-opus-4-8` and `gpt-6-luna` can switch providers without a separate
provider argument. Inference succeeds only when exactly one API provider's
catalog has the ID; a path (`/`, `./`, `../`, `~/`) selects `SelfHosted`.

This is used by model-switching tools so callers can usually pass just `model_id`.

## Context-window tags

Anthropic model IDs may include window tags such as:

```bash
sagent --provider Anthropic --model claude-sonnet-4-6+200k
sagent --provider Anthropic --model claude-opus-4-7+1m
```

The provider strips the tag for API calls and uses it to select the request-token budget.

## OpenAI-compatible endpoints

Subclass `OpenAICompat` for endpoints that implement OpenAI chat completions.

Set the subclass's `ENV_VAR`, `BASE_URL`, and `catalog`. The catalog must map
the `default` and `utility` roles to capability rows alongside its model IDs.

Use it like any other provider:

```python
model = LocalProvider.from_env().model("local-model")
```

`OpenAICompat.from_env(base_url=...)` and `from_key(api_key, base_url=...)` can override the class `BASE_URL` at construction time.

See `examples/openai_compatible_provider.py` for a runnable version.

## Self-hosted HuggingFace models

Use `SelfHosted` for HuggingFace causal LMs loaded through `transformers`:

```bash
uv sync --extra selfhosted
# Or: pip install "sagent[selfhosted]"
hf download Qwen/Qwen3.6-27B --local-dir /opt/models/qwen3.6-27b
sagent --provider SelfHosted
sagent --provider SelfHosted --model /opt/models/qwen3.6-27b+bfloat16+cuda
sagent --provider SelfHosted --model Qwen/Qwen3-0.6B+float16+cuda \
  --effort none --max-tool-call-rounds 1
```

Python API:

```python
from sagent.providers import SelfHosted, SelfHostedModel

provider = SelfHosted.from_key("Qwen/Qwen3.6-27B+bfloat16+cuda")
model: SelfHostedModel = provider.model()
```

Pass a local snapshot path to `from_key` or `--model` when you want to use an
already-populated cache. SelfHosted options such as `+cuda`, `+bfloat16`, and
`+compile` can appear in any order. Cloud Qwen IDs continue to infer
`DashScope`; select `SelfHosted` explicitly for local model paths.

Examples of frontier open-weight HuggingFace repos to evaluate:

| Model | Repo ID |
| --- | --- |
| DeepSeek V4 Flash | `deepseek-ai/DeepSeek-V4-Flash` |
| DeepSeek V4 Pro | `deepseek-ai/DeepSeek-V4-Pro` |
| Qwen 3.6 35B-A3B | `Qwen/Qwen3.6-35B-A3B` |
| Qwen 3.6 27B | `Qwen/Qwen3.6-27B` |
| Kimi K2 Thinking | `moonshotai/Kimi-K2-Thinking` |
| GLM 4.6 | `zai-org/GLM-4.6` |
| Gemma 4 31B IT | `google/gemma-4-31B-it` |

Check each model card for license, hardware, quantization, chat-template, and
`trust_remote_code` requirements. Some newly released architectures require the
latest `transformers` build or a serving runtime such as vLLM or SGLang before
they work with `AutoModelForCausalLM`; multimodal or custom-code models may need
additional runtime support.

The `selfhosted` extra tracks the released HuggingFace runtime stack needed by
these examples: `transformers`, `accelerate`, `torchvision`,
`compressed-tensors`, `sentencepiece`, `protobuf`, `safetensors`, and `torch`.
Kimi checkpoints are tagged as custom-code models, so they still require an
explicit `trust_remote_code=True` load.

## Model contract

A model exposes:

- `buffer(request)`: send one request and return a complete response.
- `stream(request, on_text=None)`: stream text chunks through `on_text` while returning a complete response.
- `is_context_overflow(error)`: classify provider context-window errors.

`ModelResponse` includes content, stop reason, token counts, response identifiers, cache counts, and cost fields.

## Pricing and costs

Provider model profiles include token limits and per-million-token pricing. Sagent records request, response, cache-write, and cache-read tokens for every response. `Agent.total_cost_usd` and CLI `--max-budget-usd` use these costs.
