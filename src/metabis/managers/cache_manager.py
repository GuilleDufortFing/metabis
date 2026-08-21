from typing import Dict, Optional


class CacheManager:
    """Manages caching for hover texts and other expensive operations."""

    def __init__(self):
        """Initialize empty cache storage."""
        self._hover_cache: Dict[str, Dict[str, str]] = {}
        self._current_cache_key: Optional[str] = None

    def get_cache_key(
        self,
        group_comparison,
        *,
        mode: str | None = None,
        active_group: str | None = None,
        activity_key: str | None = None,
    ) -> Optional[str]:
        """Create a unique key for caching hover text.

        The key must name every input the hover text depends on. In activity mode
        that is the whole activity context (group, overlay, summary, scaling), and
        nothing from the comparison, which is not shown there.

        Args:
            group_comparison: GroupComparison object or None
            mode: Current AnalysisWorkspaceMode value
            active_group: Selected group when no activity context exists yet
            activity_key: ActivityConfig.metric_key when activity mode is applied

        Returns:
            str: Cache key string, or None when nothing identifies the context
        """
        if activity_key:
            return f"mode={mode or 'experiment_analysis'}|activity={activity_key}"

        if not group_comparison:
            if active_group:
                return f"mode={mode or 'unknown'}|active_group={active_group}"
            return None

        correction = getattr(group_comparison, "correction", "unknown")
        thresholds = ()
        if hasattr(group_comparison, "get_thresholds"):
            for element_type, for_genes in (("reaction", False), ("gene", True)):
                # A reaction-only comparison names no gene thresholds, so it must
                # not contribute gene entries to the key (the live view raised
                # AttributeError here, which was skipped the same way).
                if for_genes and getattr(group_comparison, "gene_results", None) is None:
                    continue
                try:
                    values = group_comparison.get_thresholds(for_genes=for_genes)
                except AttributeError:
                    continue
                thresholds += tuple(
                    (element_type, name, value)
                    for name, value in sorted(values.items())
                )
        return "|".join(
            (
                f"mode={mode or 'group_comparison'}",
                f"membership={group_comparison.membership.fingerprint}",
                f"target_label={group_comparison.membership.target_label}",
                f"baseline_label={group_comparison.membership.baseline_label}",
                f"test={getattr(group_comparison, 'test', 'unknown')}",
                f"correction={correction}",
                f"thresholds={thresholds!r}",
            )
        )

    def get_cached_hover(self, cache_key: str, hover_key: str) -> Optional[str]:
        """Get cached hover text.

        Args:
            cache_key: Main cache key (usually based on group comparison)
            hover_key: Specific hover key (element-specific)

        Returns:
            str: Cached hover text, or None if not found
        """
        if cache_key not in self._hover_cache:
            return None

        return self._hover_cache[cache_key].get(hover_key)

    def cache_hover(self, cache_key: str, hover_key: str, hover_text: str):
        """Cache hover text.

        Args:
            cache_key: Main cache key (usually based on group comparison)
            hover_key: Specific hover key (element-specific)
            hover_text: HTML hover text to cache
        """
        if cache_key not in self._hover_cache:
            self._hover_cache[cache_key] = {}

        self._hover_cache[cache_key][hover_key] = hover_text

    def clear_cache(self):
        """Clear all caches."""
        self._hover_cache.clear()
        self._current_cache_key = None

    def clear_cache_for_key(self, cache_key: str):
        """Clear cache for a specific key.

        Args:
            cache_key: Cache key to clear
        """
        if cache_key in self._hover_cache:
            del self._hover_cache[cache_key]

    def invalidate_on_key_change(self, new_cache_key: str):
        """Handle cache invalidation when the cache key changes.

        Args:
            new_cache_key: The new cache key
        """
        if self._current_cache_key != new_cache_key:
            # Only clear if we're switching between different comparisons
            # Keep cache if just switching to/from None
            if self._current_cache_key is not None and new_cache_key is not None:
                # We're switching between different group comparisons
                pass  # Keep both caches

            self._current_cache_key = new_cache_key

    def get_cache_stats(self) -> Dict[str, int]:
        """Get statistics about current cache usage.

        Returns:
            Dict with cache statistics
        """
        total_entries = sum(len(cache) for cache in self._hover_cache.values())
        return {
            "cache_keys": len(self._hover_cache),
            "total_entries": total_entries,
            "current_key": self._current_cache_key,
        }
