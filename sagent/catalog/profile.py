"""Typed provider/model facts between generated data and runtime catalogs.

The generated module contains only normalized models.dev values.  This module
owns their typed shape; Sagent policy and transport capabilities remain outside
it.  Missing values stay ``None`` so aggregate silence is never turned into a
claim.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import date
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, cast

import tomllib

from sagent.catalog.generated.models_dev import BASELINE


if TYPE_CHECKING:
    from pathlib import Path


__all__ = [
    "CapabilityProfile",
    "FieldSource",
    "Modalities",
    "ModeCondition",
    "ModeOverlay",
    "ModelProfile",
    "PriceProfile",
    "ProfileError",
    "load_generated_profiles",
    "load_profile_augmentations",
]


type JsonValue = (
    bool | int | float | str | list[JsonValue] | dict[str, JsonValue] | None
)
type ProfileValue = bool | int | float | str | None
type ReasoningOptionType = Literal["toggle", "effort", "budget_tokens"]


class ProfileError(ValueError):
    """Generated or augmented profile data violates the declared schema."""


@dataclass(frozen=True, slots=True, kw_only=True)
class FieldSource:
    """Vendor evidence for one augmented fact or overlay."""

    url: str
    checked: date
    note: str

    def __post_init__(self) -> None:
        """Reject incomplete evidence."""
        if not self.url.startswith("https://"):
            raise ProfileError("source URL must use https")
        if not self.note.strip():
            raise ProfileError("source note must be non-empty")


@dataclass(frozen=True, slots=True, kw_only=True)
class Modalities:
    """Input and output media names reported by the aggregate."""

    input: frozenset[str] = frozenset()
    output: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True, kw_only=True)
class PriceProfile:
    """Base per-million-token prices reported for one route."""

    input: float | None = None
    output: float | None = None
    cache_read: float | None = None
    cache_write: float | None = None
    cache_write_1h: float | None = None
    raw_tiers: tuple[dict[str, JsonValue], ...] = ()
    """Normalized models.dev tiers retained until Sagent price conversion."""


@dataclass(frozen=True, slots=True, kw_only=True)
class CapabilityProfile:
    """Boolean and structured capabilities reported by models.dev."""

    attachment: bool | None = None
    reasoning: bool | None = None
    tool_call: bool | None = None
    structured_output: bool | None = None
    temperature: bool | None = None
    reasoning_options: tuple[dict[str, JsonValue], ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class ModeCondition:
    """Finite settings predicate selecting a conditional overlay."""

    context: str | None = None
    thinking: bool | None = None
    effort: str | None = None
    service_tier: str | None = None

    def __post_init__(self) -> None:
        """Require at least one supported condition."""
        if all(
            value is None
            for value in (self.context, self.thinking, self.effort, self.service_tier)
        ):
            raise ProfileError("mode overlay must have at least one condition")

    @property
    def specificity(self) -> int:
        """Number of settings axes constrained by this condition."""
        return sum(
            value is not None
            for value in (self.context, self.thinking, self.effort, self.service_tier)
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ModeOverlay:
    """Facts that differ under one declared settings condition."""

    when: ModeCondition
    context_tokens: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    prices: PriceProfile | None = None
    capabilities: CapabilityProfile | None = None
    request_defaults: dict[str, JsonValue] = field(default_factory=dict)
    source: FieldSource | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelProfile:
    """Normalized facts for one model on one provider route."""

    provider_id: str
    model_id: str
    name: str | None = None
    family: str | None = None
    status: str | None = None
    release_date: str | None = None
    last_updated: str | None = None
    knowledge: str | None = None
    open_weights: bool | None = None
    modalities: Modalities | None = None
    context_tokens: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    prices: PriceProfile | None = None
    capabilities: CapabilityProfile | None = None
    modes: tuple[ModeOverlay, ...] = ()
    unrecognized_modes: tuple[str, ...] = ()
    """Upstream mode names retained in generated data but not activated."""
    sources: dict[str, FieldSource] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze mappings and reject ambiguous conditional overlays."""
        if not self.provider_id or not self.model_id:
            raise ProfileError("provider_id and model_id must be non-empty")
        object.__setattr__(self, "sources", MappingProxyType(dict(self.sources)))
        seen: set[ModeCondition] = set()
        for overlay in self.modes:
            if overlay.when in seen:
                raise ProfileError(
                    f"duplicate mode condition for {self.provider_id}.{self.model_id}",
                )
            seen.add(overlay.when)

    def with_overlay(self, overlay: ModeOverlay) -> ModelProfile:
        """Return this profile with ``overlay`` applied to owned facts."""
        return replace(
            self,
            context_tokens=(
                overlay.context_tokens
                if overlay.context_tokens is not None
                else self.context_tokens
            ),
            input_tokens=(
                overlay.input_tokens
                if overlay.input_tokens is not None
                else self.input_tokens
            ),
            output_tokens=(
                overlay.output_tokens
                if overlay.output_tokens is not None
                else self.output_tokens
            ),
            prices=overlay.prices or self.prices,
            capabilities=overlay.capabilities or self.capabilities,
        )

    def resolve(
        self,
        *,
        context: str,
        thinking: bool,
        effort: str,
        service_tier: str,
    ) -> ModelProfile:
        """Apply every matching overlay from least to most specific.

        Two matching overlays at equal specificity are ambiguous even when
        their conditions differ: neither has precedence in the approved merge
        contract, so catalog construction must resolve the conflict instead of
        runtime order deciding it.
        """
        matches = [
            overlay
            for overlay in self.modes
            if _matches(
                overlay.when,
                context=context,
                thinking=thinking,
                effort=effort,
                service_tier=service_tier,
            )
        ]
        by_specificity: dict[int, list[ModeOverlay]] = {}
        for overlay in matches:
            by_specificity.setdefault(overlay.when.specificity, []).append(overlay)
        conflicts = [items for items in by_specificity.values() if len(items) > 1]
        if conflicts:
            raise ProfileError(
                f"equal-specificity overlays match {self.provider_id}.{self.model_id}",
            )
        resolved = self
        for overlay in sorted(matches, key=lambda item: item.when.specificity):
            resolved = resolved.with_overlay(overlay)
        return resolved


def _matches(
    condition: ModeCondition,
    *,
    context: str,
    thinking: bool,
    effort: str,
    service_tier: str,
) -> bool:
    """Whether one finite condition matches selected settings."""
    return all(
        expected is None or expected == actual
        for expected, actual in (
            (condition.context, context),
            (condition.thinking, thinking),
            (condition.effort, effort),
            (condition.service_tier, service_tier),
        )
    )


def load_generated_profiles(
    baseline: Mapping[str, object] = BASELINE,
) -> Mapping[str, Mapping[str, ModelProfile]]:
    """Parse the checked-in aggregate into typed profiles.

    Only structured, provider-verified service-tier modes are activated.
    Every other upstream mode remains named by ``unrecognized_modes`` for a
    refresh report or maintainer review.
    """
    providers = _mapping(baseline.get("providers"), "providers")
    parsed: dict[str, Mapping[str, ModelProfile]] = {}
    for provider_id, raw_provider in providers.items():
        provider = _mapping(raw_provider, provider_id)
        models = _mapping(provider.get("models"), f"{provider_id}.models")
        parsed[provider_id] = MappingProxyType(
            {
                model_id: _profile(provider_id, model_id, raw_model)
                for model_id, raw_model in models.items()
            },
        )
    return MappingProxyType(parsed)


def _profile(provider_id: str, model_id: str, raw: object) -> ModelProfile:
    model = _mapping(raw, f"{provider_id}.{model_id}")
    if model.get("id") != model_id:
        raise ProfileError(f"{provider_id}.{model_id} reports a different id")
    limit = _optional_mapping(model.get("limit"), f"{provider_id}.{model_id}.limit")
    modalities = _modalities(model.get("modalities"), provider_id, model_id)
    capabilities = CapabilityProfile(
        attachment=_optional_bool(model.get("attachment"), "attachment"),
        reasoning=_optional_bool(model.get("reasoning"), "reasoning"),
        tool_call=_optional_bool(model.get("tool_call"), "tool_call"),
        structured_output=_optional_bool(
            model.get("structured_output"), "structured_output"
        ),
        temperature=_optional_bool(model.get("temperature"), "temperature"),
        reasoning_options=_json_dicts(model.get("reasoning_options")),
    )
    modes, unknown = _modes(provider_id, model_id, model.get("experimental"))
    return ModelProfile(
        provider_id=provider_id,
        model_id=model_id,
        name=_optional_str(model.get("name"), "name"),
        family=_optional_str(model.get("family"), "family"),
        status=_optional_str(model.get("status"), "status"),
        release_date=_optional_str(model.get("release_date"), "release_date"),
        last_updated=_optional_str(model.get("last_updated"), "last_updated"),
        knowledge=_optional_str(model.get("knowledge"), "knowledge"),
        open_weights=_optional_bool(model.get("open_weights"), "open_weights"),
        modalities=modalities,
        context_tokens=_optional_int(limit.get("context"), "limit.context"),
        input_tokens=_optional_int(limit.get("input"), "limit.input"),
        output_tokens=_optional_int(limit.get("output"), "limit.output"),
        prices=_price(model.get("cost"), f"{provider_id}.{model_id}.cost"),
        capabilities=capabilities,
        modes=modes,
        unrecognized_modes=unknown,
    )


def _modes(
    provider_id: str,
    model_id: str,
    experimental_raw: object,
) -> tuple[tuple[ModeOverlay, ...], tuple[str, ...]]:
    if experimental_raw is None:
        return (), ()
    experimental = _mapping(experimental_raw, f"{provider_id}.{model_id}.experimental")
    raw_modes = experimental.get("modes")
    if raw_modes is None:
        return (), ()
    modes = _mapping(raw_modes, f"{provider_id}.{model_id}.experimental.modes")
    recognized: list[ModeOverlay] = []
    unknown: list[str] = []
    for name, raw_mode in sorted(modes.items()):
        mode = _mapping(raw_mode, f"{provider_id}.{model_id}.mode.{name}")
        tier = _service_tier(provider_id, mode)
        if tier is None:
            unknown.append(name)
            continue
        recognized.append(
            ModeOverlay(
                when=ModeCondition(service_tier=tier),
                prices=_price(
                    mode.get("cost"), f"{provider_id}.{model_id}.{name}.cost"
                ),
            ),
        )
    return tuple(recognized), tuple(unknown)


def _service_tier(provider_id: str, mode: Mapping[str, object]) -> str | None:
    """Map only exact request shapes verified against provider documentation."""
    provider = _optional_mapping(mode.get("provider"), "mode.provider")
    body = _optional_mapping(provider.get("body"), "mode.provider.body")
    headers = _optional_mapping(provider.get("headers"), "mode.provider.headers")
    if provider_id == "openai" and not headers and set(body) == {"service_tier"}:
        value = body["service_tier"]
        if value in {"fast", "priority"}:
            return "priority"
        if value in {"flex", "ultrafast"}:
            return cast(str, value)
        return None
    if (
        provider_id == "anthropic"
        and body == {"speed": "fast"}
        and (not headers or headers == {"anthropic-beta": "fast-mode-2026-02-01"})
    ):
        return "priority"
    return None


def _price(raw: object, path: str) -> PriceProfile | None:
    if raw is None:
        return None
    cost = _mapping(raw, path)
    tiers_raw = cost.get("tiers")
    tiers = _json_dicts(tiers_raw)
    return PriceProfile(
        input=_optional_number(cost.get("input"), f"{path}.input"),
        output=_optional_number(cost.get("output"), f"{path}.output"),
        cache_read=_optional_number(cost.get("cache_read"), f"{path}.cache_read"),
        cache_write=_optional_number(cost.get("cache_write"), f"{path}.cache_write"),
        cache_write_1h=_optional_number(
            cost.get("cache_write_1h"), f"{path}.cache_write_1h"
        ),
        raw_tiers=tiers,
    )


def _modalities(raw: object, provider_id: str, model_id: str) -> Modalities | None:
    if raw is None:
        return None
    value = _mapping(raw, f"{provider_id}.{model_id}.modalities")
    return Modalities(
        input=_string_set(value.get("input"), "modalities.input"),
        output=_string_set(value.get("output"), "modalities.output"),
    )


def _mapping(raw: object, path: str) -> Mapping[str, object]:
    if not isinstance(raw, Mapping):
        raise ProfileError(f"{path} must be an object")
    return cast(Mapping[str, object], raw)


def _optional_mapping(raw: object, path: str) -> Mapping[str, object]:
    return {} if raw is None else _mapping(raw, path)


def _optional_str(raw: object, path: str) -> str | None:
    if raw is None or isinstance(raw, str):
        return raw
    raise ProfileError(f"{path} must be a string")


def _optional_bool(raw: object, path: str) -> bool | None:
    if raw is None or isinstance(raw, bool):
        return raw
    raise ProfileError(f"{path} must be a boolean")


def _optional_int(raw: object, path: str) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, int) and not isinstance(raw, bool):
        return raw
    raise ProfileError(f"{path} must be an integer")


def _optional_number(raw: object, path: str) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return float(raw)
    raise ProfileError(f"{path} must be a number")


def _string_set(raw: object, path: str) -> frozenset[str]:
    if raw is None:
        return frozenset()
    if not isinstance(raw, list):
        raise ProfileError(f"{path} must be an array of strings")
    items = cast(list[object], raw)
    if not all(isinstance(item, str) for item in items):
        raise ProfileError(f"{path} must be an array of strings")
    return frozenset(cast(list[str], items))


def _json_dicts(raw: object) -> tuple[dict[str, JsonValue], ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ProfileError("structured options must be an array of objects")
    items = cast(list[object], raw)
    if not all(isinstance(item, dict) for item in items):
        raise ProfileError("structured options must be an array of objects")
    return tuple(cast(list[dict[str, JsonValue]], items))


_PROFILE_SCALARS = frozenset(
    {
        "name",
        "family",
        "status",
        "release_date",
        "last_updated",
        "knowledge",
        "open_weights",
        "context_tokens",
        "input_tokens",
        "output_tokens",
    }
)
_METADATA = frozenset({"source_url", "checked", "note", "augmentation_only"})
_CONDITIONS = frozenset(
    {"when_context", "when_thinking", "when_effort", "when_service_tier"}
)


def load_profile_augmentations(
    path: Path,
    profiles: Mapping[str, Mapping[str, ModelProfile]],
) -> Mapping[str, Mapping[str, ModelProfile]]:
    """Load and apply reviewed TOML corrections with mandatory provenance."""
    try:
        decoded = tomllib.loads(path.read_text())
    except (OSError, tomllib.TOMLDecodeError) as e:
        raise ProfileError(f"cannot load augmentations from {path}: {e}") from e
    unknown_top = set(decoded) - {"provider"}
    if unknown_top:
        raise ProfileError(f"unknown augmentation sections: {sorted(unknown_top)!r}")
    raw_providers = _mapping(decoded.get("provider", {}), "provider")
    result: dict[str, dict[str, ModelProfile]] = {
        provider_id: dict(models) for provider_id, models in profiles.items()
    }
    for provider_id, raw_provider in raw_providers.items():
        if provider_id not in result:
            raise ProfileError(f"augmentation names unknown provider {provider_id!r}")
        provider = _mapping(raw_provider, f"provider.{provider_id}")
        unknown_provider = set(provider) - {"source_url", "checked", "model"}
        if unknown_provider:
            raise ProfileError(
                f"unknown fields in provider.{provider_id}: {sorted(unknown_provider)!r}"
            )
        inherited_url = _required_str(provider, "source_url", f"provider.{provider_id}")
        inherited_checked = _required_date(
            provider, "checked", f"provider.{provider_id}"
        )
        raw_models = _mapping(
            provider.get("model", {}), f"provider.{provider_id}.model"
        )
        for model_id, raw_model in raw_models.items():
            model = _mapping(raw_model, f"provider.{provider_id}.model.{model_id}")
            augmentation_only = model.get("augmentation_only", False)
            if not isinstance(augmentation_only, bool):
                raise ProfileError(
                    f"{provider_id}.{model_id}.augmentation_only must be boolean"
                )
            current = result[provider_id].get(model_id)
            if current is None:
                if not augmentation_only:
                    raise ProfileError(
                        f"augmentation names unknown model {provider_id}.{model_id}"
                    )
                current = ModelProfile(provider_id=provider_id, model_id=model_id)
            result[provider_id][model_id] = _augment_model(
                current,
                model,
                inherited_url=inherited_url,
                inherited_checked=inherited_checked,
            )
    return MappingProxyType(
        {
            provider_id: MappingProxyType(models)
            for provider_id, models in result.items()
        }
    )


def _augment_model(
    profile: ModelProfile,
    raw: Mapping[str, object],
    *,
    inherited_url: str,
    inherited_checked: date,
) -> ModelProfile:
    path = f"{profile.provider_id}.{profile.model_id}"
    allowed = (
        _PROFILE_SCALARS | _METADATA | {"modalities", "prices", "capabilities", "mode"}
    )
    unknown = set(raw) - allowed
    if unknown:
        raise ProfileError(f"unknown fields in {path}: {sorted(unknown)!r}")
    changed = set(raw) & (_PROFILE_SCALARS | {"modalities", "prices", "capabilities"})
    model_url = raw.get("source_url", inherited_url)
    model_checked = raw.get("checked", inherited_checked)
    if not isinstance(model_url, str) or not isinstance(model_checked, date):
        raise ProfileError(f"{path} has malformed source metadata")
    source = (
        _correction_source(
            raw,
            path,
            inherited_url=model_url,
            inherited_checked=model_checked,
        )
        if changed
        else None
    )
    updates: dict[str, object] = {}
    for field_name in changed & _PROFILE_SCALARS:
        updates[field_name] = _profile_scalar(field_name, raw[field_name], path)
    if "modalities" in changed:
        updates["modalities"] = _modalities(
            raw["modalities"], profile.provider_id, profile.model_id
        )
    if "prices" in changed:
        updates["prices"] = _price(raw["prices"], f"{path}.prices")
    if "capabilities" in changed:
        values = _mapping(raw["capabilities"], f"{path}.capabilities")
        allowed_capabilities = {
            "attachment",
            "reasoning",
            "tool_call",
            "structured_output",
            "temperature",
        }
        if set(values) - allowed_capabilities:
            raise ProfileError(f"unknown capability fields in {path}")
        updates["capabilities"] = _capability_profile(
            values,
            f"{path}.capabilities",
        )
    sources = dict(profile.sources)
    if source is not None:
        sources.update(dict.fromkeys(changed, source))
    modes = list(profile.modes)
    raw_modes = _mapping(raw.get("mode", {}), f"{path}.mode")
    for mode_name, raw_mode in raw_modes.items():
        mode = _mapping(raw_mode, f"{path}.mode.{mode_name}")
        overlay = _augmentation_mode(
            mode,
            f"{path}.mode.{mode_name}",
            inherited_url=model_url,
            inherited_checked=model_checked,
        )
        modes = [existing for existing in modes if existing.when != overlay.when]
        modes.append(overlay)
    return replace(profile, **updates, modes=tuple(modes), sources=sources)


def _augmentation_mode(
    raw: Mapping[str, object],
    path: str,
    *,
    inherited_url: str,
    inherited_checked: date,
) -> ModeOverlay:
    allowed = (
        _CONDITIONS
        | _METADATA
        | {"context_tokens", "input_tokens", "output_tokens", "prices", "capabilities"}
    )
    unknown = set(raw) - allowed
    if unknown:
        raise ProfileError(f"unknown fields in {path}: {sorted(unknown)!r}")
    source = _correction_source(
        raw,
        path,
        inherited_url=inherited_url,
        inherited_checked=inherited_checked,
    )
    condition = ModeCondition(
        context=_optional_str(raw.get("when_context"), f"{path}.when_context"),
        thinking=_optional_bool(raw.get("when_thinking"), f"{path}.when_thinking"),
        effort=_optional_str(raw.get("when_effort"), f"{path}.when_effort"),
        service_tier=_optional_str(
            raw.get("when_service_tier"), f"{path}.when_service_tier"
        ),
    )
    raw_capabilities = raw.get("capabilities")
    capabilities = (
        _capability_profile(
            _mapping(raw_capabilities, f"{path}.capabilities"),
            f"{path}.capabilities",
        )
        if raw_capabilities is not None
        else None
    )
    return ModeOverlay(
        when=condition,
        context_tokens=_optional_int(
            raw.get("context_tokens"), f"{path}.context_tokens"
        ),
        input_tokens=_optional_int(raw.get("input_tokens"), f"{path}.input_tokens"),
        output_tokens=_optional_int(raw.get("output_tokens"), f"{path}.output_tokens"),
        prices=_price(raw.get("prices"), f"{path}.prices"),
        capabilities=capabilities,
        source=source,
    )


def _correction_source(
    raw: Mapping[str, object],
    path: str,
    *,
    inherited_url: str,
    inherited_checked: date,
) -> FieldSource:
    url = raw.get("source_url", inherited_url)
    checked = raw.get("checked", inherited_checked)
    note = raw.get("note")
    if not isinstance(url, str) or not isinstance(checked, date):
        raise ProfileError(f"{path} has malformed source metadata")
    if not isinstance(note, str):
        raise ProfileError(f"{path} correction requires a factual note")
    return FieldSource(url=url, checked=checked, note=note)


def _profile_scalar(name: str, raw: object, path: str) -> object:
    if name in {"context_tokens", "input_tokens", "output_tokens"}:
        value = _optional_int(raw, f"{path}.{name}")
        if value is None:
            raise ProfileError(f"{path}.{name} cannot clear a fact")
        return value
    if name == "open_weights":
        value = _optional_bool(raw, f"{path}.{name}")
        if value is None:
            raise ProfileError(f"{path}.{name} cannot clear a fact")
        return value
    value = _optional_str(raw, f"{path}.{name}")
    if value is None:
        raise ProfileError(f"{path}.{name} cannot clear a fact")
    return value


def _capability_profile(values: Mapping[str, object], path: str) -> CapabilityProfile:
    allowed = {
        "attachment",
        "reasoning",
        "tool_call",
        "structured_output",
        "temperature",
    }
    unknown = set(values) - allowed
    if unknown:
        raise ProfileError(f"unknown capability fields in {path}: {sorted(unknown)!r}")
    return CapabilityProfile(
        attachment=_optional_bool(values.get("attachment"), f"{path}.attachment"),
        reasoning=_optional_bool(values.get("reasoning"), f"{path}.reasoning"),
        tool_call=_optional_bool(values.get("tool_call"), f"{path}.tool_call"),
        structured_output=_optional_bool(
            values.get("structured_output"), f"{path}.structured_output"
        ),
        temperature=_optional_bool(values.get("temperature"), f"{path}.temperature"),
    )


def _required_str(raw: Mapping[str, object], name: str, path: str) -> str:
    value = raw.get(name)
    if not isinstance(value, str):
        raise ProfileError(f"{path}.{name} is required")
    return value


def _required_date(raw: Mapping[str, object], name: str, path: str) -> date:
    value = raw.get(name)
    if not isinstance(value, date):
        raise ProfileError(f"{path}.{name} must be a TOML date")
    return value
