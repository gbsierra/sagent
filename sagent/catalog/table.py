"""Which model a name means, and what it can do over one transport.

A vendor catalog is a :class:`ModelTable`: its rows, each keyed by the id the
vendor's API takes and reports back (``claude-opus-5-5``, ``gpt-5.6-sol``,
``MiniMax-M3``), plus three explicit name maps. One name reaches a row four
ways, in this order:

1. its vendor id (``claude-opus-5-5``). A vendor alias and the snapshot it
   points at are both vendor ids, so each is its own row carrying the same
   facts (``claude-haiku-4-5`` and ``claude-haiku-4-5-20251001``): the
   request sends exactly what was named, and a response reporting either
   prices at its own row;
2. a Sagent role (``default``, ``utility``, ``best``);
3. a current Sagent convenience alias (``sol``);
4. a retired Sagent spelling (``opus-5.5``), so sessions and settings saved
   before rows were keyed by vendor id still load. Resolution returns the
   vendor id, so a retired spelling is translated once and never saved again.

Nothing is inferred from how an id is spelled. Vendors share no grammar
(``claude-opus-4-5``, ``gpt-5.4-mini``, ``qwen3.6-max-preview``,
``MiniMax-M2.7``), and which id is "newest" or what an alias points at is a
fact only the vendor can state: a table that guessed it from a prefix and row
order silently moved every role when a row was added. A name that reaches no
row is unknown, and the error lists what is known.

:class:`ModelCatalog` meets a table with one transport and resolves a
possibly context-tagged name (``claude-opus-5-5+200k``) to a capability and
the settings that name selected.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Final, cast, get_args, override

import logging

from sagent.catalog.provenance import Fact, Source
from sagent.types.capability import (
    ContextTag,
    ModelCapability,
    ModelSettings,
)


__all__ = [
    "CONTEXT_TAGS",
    "ModelCatalog",
    "ModelTable",
    "UnknownModelError",
    "UnsupportedTagError",
    "base_model_id",
    "split_model_id",
]

logger = logging.getLogger(__name__)

# Every response of a session reports the same served id, so one warning per id
# says it; one per response would bury the log.
_unlisted_served: set[str] = set()


# Derived from ``ContextTag``, not restated: a tag the type admits but a
# hand-written tuple omitted would be unparseable while type-checking clean.
# The default window's ``""`` is filtered out -- no id carries it.
CONTEXT_TAGS: Final[tuple[ContextTag, ...]] = tuple(
    t
    for t in cast(tuple[ContextTag, ...], get_args(cast(object, ContextTag.__value__)))
    if t
)
"""Window-size suffixes a model id may carry (e.g. ``...+1m``)."""


class UnknownModelError(ValueError):
    """No row answers to the name."""


class UnsupportedTagError(ValueError):
    """The name carries a context tag the model does not offer."""


def split_model_id(model_id: str) -> tuple[str, frozenset[ContextTag]]:
    """Split a model id into its base id and trailing context tags.

    Tags may appear in any order; matching is case-insensitive, and an
    unknown suffix stays part of the base id.

    Args:
      model_id: Model id, possibly with trailing context tags.

    Returns:
      base_id: ``model_id`` without its tags.
      tags: The stripped tags, lowercased (e.g. ``{"+1m"}``).

    """
    tags: set[ContextTag] = set()
    base = model_id
    while True:
        lower = base.lower()
        tag: ContextTag | None = next(
            (t for t in CONTEXT_TAGS if lower.endswith(t)),
            None,
        )
        if tag is None:
            return base, frozenset(tags)
        tags.add(tag)
        base = base[: -len(tag)]


def base_model_id(model_id: str) -> str:
    """Strip trailing context tags, yielding the canonical model id."""
    return split_model_id(model_id)[0]


def _frozen() -> Mapping[str, str]:
    return MappingProxyType(dict[str, str]())


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelTable(Mapping[str, ModelCapability]):
    """Vendor rows reachable by exact id, role, alias, or retired name.

    Iterating yields vendor ids only: name maps are ways to select a row, not
    additional models.
    """

    rows: tuple[ModelCapability, ...]
    """Every row; order carries no meaning."""

    roles: Mapping[str, str] = field(default_factory=_frozen)
    """Sagent role to the vendor id it means (``{"default": "claude-opus-5-5"}``)."""

    aliases: Mapping[str, str] = field(default_factory=_frozen)
    """Current convenience name to an exact vendor id (``{"sol": "gpt-6.1-sol"}``)."""

    renamed: Mapping[str, str] = field(default_factory=_frozen)
    """Retired Sagent spelling to its vendor id; accepted, never produced."""

    sources: Mapping[Fact, Source] = field(
        default_factory=lambda: MappingProxyType(dict[Fact, Source]()),
    )
    """Where every row's facts were read, per fact group."""

    row_sources: Mapping[str, Mapping[Fact, Source]] = field(
        default_factory=lambda: MappingProxyType(dict[str, Mapping[Fact, Source]]()),
    )
    """Vendor id to the groups its row read from a page other than ``sources``."""

    _by_id: Mapping[str, ModelCapability] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Index rows; check each name points at a row and shadows nothing."""
        by_id: dict[str, ModelCapability] = {}
        for row in self.rows:
            if row.model_id in by_id:
                raise ValueError(f"duplicate model id {row.model_id!r}")
            by_id[row.model_id] = row
        object.__setattr__(self, "_by_id", MappingProxyType(by_id))
        object.__setattr__(self, "sources", MappingProxyType(dict(self.sources)))
        stray = sorted(self.row_sources.keys() - by_id.keys())
        if stray:
            raise ValueError(f"sources for no model: {', '.join(stray)}")
        object.__setattr__(
            self,
            "row_sources",
            MappingProxyType(
                {mid: MappingProxyType(dict(s)) for mid, s in self.row_sources.items()},
            ),
        )
        roles = MappingProxyType(dict(self.roles))
        aliases = MappingProxyType(dict(self.aliases))
        renamed = MappingProxyType(dict(self.renamed))
        object.__setattr__(self, "roles", roles)
        object.__setattr__(self, "aliases", aliases)
        object.__setattr__(self, "renamed", renamed)
        name_maps = (
            ("role", roles),
            ("alias", aliases),
            ("retired name", renamed),
        )
        for kind, names in name_maps:
            for name, target in names.items():
                if target not in by_id:
                    raise ValueError(f"{kind} {name!r} names no model: {target!r}")
                if name in by_id:
                    raise ValueError(f"{kind} {name!r} shadows a model id")
        for i, (left_kind, left) in enumerate(name_maps):
            for right_kind, right in name_maps[i + 1 :]:
                clash = sorted(left.keys() & right.keys())
                if clash:
                    left_article = "an" if left_kind == "alias" else "a"
                    right_article = "an" if right_kind == "alias" else "a"
                    raise ValueError(
                        f"both {left_article} {left_kind} and"
                        f" {right_article} {right_kind}: {', '.join(clash)}",
                    )

    @override
    def __getitem__(self, key: str) -> ModelCapability:
        row = self._named(key)
        if row is None:
            raise KeyError(key)
        return row

    @override
    def __iter__(self) -> Iterator[str]:
        return iter(self._by_id)

    @override
    def __len__(self) -> int:
        return len(self._by_id)

    @override
    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and self._named(key) is not None

    def exact(self, model_id: str) -> ModelCapability | None:
        """Return the row whose vendor id is ``model_id``, never via a name map.

        For ids a server reports back: a role or retired name is Sagent's
        spelling, and no server reports it.

        Args:
          model_id: Vendor id.

        Returns:
          row: The row, or ``None`` when no row carries that id.

        """
        return self._by_id.get(model_id)

    def source(self, model_id: str, fact: Fact) -> Source | None:
        """Return where the row with vendor id ``model_id`` read ``fact``.

        Args:
          model_id: Vendor id.
          fact: The fact group asked about.

        Returns:
          source: The row's own source for ``fact``, else the table's; ``None``
            when neither names one.

        """
        own = self.row_sources.get(model_id)
        if own is not None and fact in own:
            return own[fact]
        return self.sources.get(fact)

    def _named(self, key: str) -> ModelCapability | None:
        """Return the row ``key`` names in the documented precedence order."""
        target = (
            key
            if key in self._by_id
            else self.roles.get(key, self.aliases.get(key, self.renamed.get(key)))
        )
        return self._by_id.get(target) if target is not None else None


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelCatalog:
    """A vendor table met with one transport."""

    models: ModelTable
    """The rows and names the transport serves."""

    transport: ModelCapability
    """What the transport allows; met with every row it resolves."""

    def served(
        self,
        requested: ModelCapability,
        served_model_id: str,
    ) -> ModelCapability:
        """Return the row a response bills at: the one the server says it ran.

        A response names the snapshot an alias resolved to, or another model
        the vendor fell back to, and that row's prices are the ones charged.
        Only an exact vendor id of THIS catalog counts: a server never reports
        a Sagent role, and two transports of one vendor may list different
        rows for the same id.

        Args:
          requested: The capability the request was built from.
          served_model_id: The id the response reports; blank when it reports
            none.

        Returns:
          capability: The served row met with this transport, or ``requested``
            when the response named none, named the requested row, or named
            an id no row lists (logged, since the bill then rests on a row
            the server did not run).

        """
        if not served_model_id or served_model_id == requested.model_id:
            return requested
        row = self.models.exact(served_model_id)
        if row is None:
            if served_model_id not in _unlisted_served:
                _unlisted_served.add(served_model_id)
                logger.warning(
                    "server ran %r, which no catalog row lists; "
                    "pricing at the requested %r",
                    served_model_id,
                    requested.model_id,
                )
            return requested
        return row & self.transport

    def resolve(self, model_id: str) -> tuple[ModelCapability, ModelSettings]:
        """Resolve a possibly context-tagged name against this transport.

        Args:
          model_id: Vendor id, role, alias, or retired name, optionally tagged.

        Returns:
          capability: The row narrowed by the transport; its ``model_id`` is
            the vendor id, whichever name reached it.
          settings: Settings selecting the requested context.

        Raises:
          UnknownModelError: No row answers to the name.
          UnsupportedTagError: Context tags conflict or are unavailable.

        """
        base, id_tags = split_model_id(model_id)
        tags: list[ContextTag] = sorted(id_tags)
        if len(tags) > 1:
            raise UnsupportedTagError(
                f"Model {model_id!r} selects conflicting contexts: {', '.join(tags)}",
            )
        context: ContextTag = tags[0] if tags else ""
        row = self.models.get(base)
        if row is None:
            lines = [f"Unknown model {model_id!r}."]
            for heading, names in (
                ("Aliases", self.models.aliases),
                ("Roles", self.models.roles),
            ):
                if names:
                    lines.extend(
                        [
                            "",
                            f"{heading}:",
                            *[f"  {k} -> {v}" for k, v in names.items()],
                        ],
                    )
            known = ", ".join(self.models) or "none"
            lines.extend(["", "Known models:", f"  {known}"])
            raise UnknownModelError("\n".join(lines))
        capability = row & self.transport
        # A window the default context already spans needs no row of its own.
        if (
            context == "+1m"
            and context not in capability.context
            and capability.context[""].max_request_tokens >= 1_000_000
        ):
            capability = replace(
                capability,
                context=MappingProxyType(
                    {**capability.context, "+1m": capability.context[""]},
                ),
            )
        if context not in capability.context:
            offered = ", ".join(sorted(t for t in capability.context if t)) or "(none)"
            raise UnsupportedTagError(
                f"Model {model_id!r} has no {context} context; offers: {offered}",
            )
        return capability, ModelSettings.narrowest(capability, context=context)
