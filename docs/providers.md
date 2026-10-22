# Providers

Sagent separates providers from models. A provider owns authentication and creates model objects. A model sends typed requests and returns typed responses.

## Public provider matrix

| Provider class | Environment variable | Default model | Utility model | Notes |
| --- | --- | --- | --- | --- |
| `Anthropic` | `ANTHROPIC_API_KEY` | `opus-5.5` | `sonnet-5.5` | Anthropic API-key provider; `best` resolves to `fable-5.1`. |
| `AnthropicCLI` | native `claude` login | `opus-5.5` | `sonnet-5.5` | Claude Code subscription through `claude`; named accounts use `CLAUDE_CONFIG_DIR`. |
| `OpenAI` | `OPENAI_API_KEY` | `sol-6.1` | `luna-6.0` | OpenAI API provider. |
| `OpenAISubscription` | Sign in with ChatGPT | `sol-6.1` | `luna-6.0` | ChatGPT plan through OpenAI's official SIWC flow; named accounts keep one sign-in record each. |
| `Google` | `GOOGLE_API_KEY` | `gemini-pro-3.1` | `gemini-flash-lite-3.5` | Google Gemini provider. |
| `GoogleCLI` | native `gemini` login | `gemini-pro-3.1` | `gemini-flash-lite-3.5` | Gemini subscription through Gemini CLI; named accounts use `GEMINI_CLI_HOME`. |
| `Moonshot` | `MOONSHOT_API_KEY` | `kimi-3.0` | `kimi-2.6` | OpenAI-compatible Kimi provider. |
| `DashScope` | `DASHSCOPE_API_KEY` | `qwen-plus-3.7` | `qwen-flash-3.8` | Alibaba DashScope provider. |
| `MiniMax` | `MINIMAX_API_KEY` | `minimax-3.0` | `minimax-3.0` | MiniMax provider. |
| `SelfHosted` | none | `Qwen/Qwen3.6-27B` | configured snapshot | Local HF transformers provider. |
| `OpenAICompat` | subclass-defined | subclass-defined | subclass-defined | Base class for chat-completions-compatible APIs. |

The public package is designed around API-key providers.

## Vendor CLI authentication

`AnthropicCLI` and `GoogleCLI` delegate authentication to the installed
vendor CLI instead of managing a live token or refresh lifecycle:

```bash
sagent --provider AnthropicCLI login
sagent --provider GoogleCLI login
```

Pass `--account NAME` to maintain a separate named login. The default account
uses the vendor's existing configuration. Named Anthropic accounts use a
sagent-owned `CLAUDE_CONFIG_DIR`; named Google accounts use a sagent-owned
`GEMINI_CLI_HOME`. The CLI providers preserve the operator's `HOME` for native
tools. GoogleCLI never copies OAuth files into temporary subprocess
directories; AnthropicCLI only reads and validates a preexisting legacy file
when migrating an older named account.

## ChatGPT subscription authentication

`OpenAISubscription` signs in with OpenAI's official "Sign in with
ChatGPT" flow. Sagent is its own client: on first sign-in it registers
with OpenAI, then it verifies the identity token OpenAI returns and
stores one sign-in record per account in Sagent's data directory. It
never borrows another tool's ChatGPT login.

```bash
sagent --provider OpenAISubscription login
sagent --provider OpenAISubscription --account work login
```

Sagent prints a sign-in URL, you finish the sign-in in your browser, and
Sagent waits for the redirect. Add `--headless` to paste the final URL
back by hand instead. The sign-in page asks for ChatGPT plan usage; if
you decline, the sign-in is saved but cannot drive inference until you
sign in again and accept. Signing out through the Python API
(`OpenAISubscription.logout`) revokes the refresh token with OpenAI and
clears the local tokens while keeping the registration, so signing back
in does not start a new one.

## Basic usage

```python
from sagent.providers import Google

provider = Google.from_env()
model = provider.model("gemini-pro-3.1")
utility = provider.model("utility")
```

All public API-key providers support:

```python
ProviderClass.from_env()
ProviderClass.from_key("...")
ProviderClass.from_env().model("model-id")
ProviderClass.from_env().model("utility")
```

`model(None)` uses the provider's default model. Unknown model IDs raise with the provider's known model list.

Catalog keys are exact names of the form `family-major.minor`, such as
`sol-6.1`, `luna-6.0`, and `opus-5.5`. A prefix names the newest model it
starts: `sol` and `sol-6` both mean `sol-6.1`, and `opus-4` means `opus-4.8`.
The roles `default`, `utility`, and `best` resolve the same way. Vendor wire
IDs remain accepted as compatibility aliases. A bare model ID selects its
largest context window. Append `+272k` for a smaller
OpenAI window or `+200k` for a smaller Anthropic window when the model offers
one. An explicit `+1m` is also accepted when the bare model already provides
that window. Both OpenAI providers use the Responses API. Supported reasoning
efforts vary by model.

API-key requests send output-token limits and temperature for non-reasoning
models. The subscription backend omits those unsupported fields. Both paths
replay local conversation history with `store: false`, preserving encrypted
reasoning and image-bearing tool results. Existing sessions need no conversion.

## CLI dispatch

```bash
sagent --provider Google --auth env --model gemini-pro-3.1
```

`--provider` is the provider class name from `sagent.providers`. `--auth env` calls `Google.from_env()`.

If the named factory does not exist, Sagent reports an error. Use `--auth env`
for API keys so secrets do not land in shell history.

## Provider inference

Agent tools first infer a provider from catalog membership, so short IDs such
as `opus-4.8` and `luna-6.0` can switch providers without a separate provider
argument. Vendor prefixes remain as compatibility fallbacks:

| Prefix | Provider |
| --- | --- |
| `claude` | `Anthropic` |
| `gemini` | `Google` |
| `gpt`, `chatgpt`, `o1`, `o3`, `o4`, `codex` | `OpenAI` |
| `kimi`, `moonshot` | `Moonshot` |
| `qwen` | `DashScope` |
| `minimax` | `MiniMax` |
| `/`, `./`, `../`, `~/` | `SelfHosted` |

This is used by model-switching tools so callers can usually pass just `model_id`.

## Context-window tags

Anthropic model IDs may include window tags such as:

```bash
sagent --provider Anthropic --model sonnet-4.6+200k
sagent --provider Anthropic --model opus-4.7+1m
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
