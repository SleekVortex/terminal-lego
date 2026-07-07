from __future__ import annotations

from sources.stackoverflow import tag_taxonomy as taxonomy


def test_normalize_tag_strips_and_lowercases() -> None:
    assert taxonomy.normalize_tag("  PyThOn  ") == "python"


def test_classify_tags_returns_terminal_bench_categories_in_order() -> None:
    categories = taxonomy.classify_tags(["bash", "openssl", "json"])

    assert "system-administration" in categories
    assert "security" in categories
    assert "data-processing" in categories
    assert categories == [c for c in taxonomy.CATEGORY_ORDER if c in categories]


def test_category_tag_hits_returns_only_matching_tags() -> None:
    hits = taxonomy.category_tag_hits(["Bash", "ssl", "unknown-tag"])

    assert hits["system-administration"] == ["bash"]
    assert hits["security"] == ["ssl"]
    assert "unknown-tag" not in {tag for tags in hits.values() for tag in tags}


def test_scaled_quotas_exact_total_and_zero_total() -> None:
    quotas = taxonomy.scaled_quotas(
        7,
        {"software-engineering": 26, "system-administration": 9, "security": 8},
    )

    assert sum(quotas.values()) == 7
    assert quotas["software-engineering"] >= quotas["security"]
    assert taxonomy.scaled_quotas(0, {"a": 1, "b": 2}) == {"a": 0, "b": 0}
