def test_support_dir_line_number_suffix_is_stripped_not_fatal():
    """``see scripts/eval.ts:8`` is a file:line citation, not a bundle path: the
    trailing ``:LINE`` / ``:LINE:COL`` suffix must be stripped instead of poisoning
    the whole install with a misleading stale-index error (#149748)."""
    from tools.skills_hub_models import _referenced_support_paths

    skill = "---\nname: x\ndescription: x\n---\n"
    assert _referenced_support_paths(
        skill + "See scripts/eval.ts:8 for the loop.\n") == {"scripts/eval.ts"}
    assert _referenced_support_paths(
        skill + "See scripts/eval.ts:8:20 for the loop.\n") == {"scripts/eval.ts"}
    # Plain paths keep working; the real bundle path still resolves alongside the citation.
    assert _referenced_support_paths(
        skill + "See scripts/eval.ts for the loop.\n") == {"scripts/eval.ts"}
    assert _referenced_support_paths(
        skill + "Run [the script](scripts/eval.ts) and see scripts/eval.ts:8.\n"
    ) == {"scripts/eval.ts"}


def test_support_dir_line_suffix_stripping_keeps_guards():
    """Stripping the line-number suffix must not weaken the traversal/glob guards."""
    from tools.skills_hub_models import _referenced_support_paths

    skill = "---\nname: x\ndescription: x\n---\n"
    # Traversal is still refused outright (fail closed).
    assert _referenced_support_paths(
        skill + "see scripts/../../etc/passwd:8\n") is None
    # Glob prose is still skipped.
    assert _referenced_support_paths(
        skill + "see scripts/*.ts:8\n") == set()
    # A trailing colon without digits was already stripped pre-#149748; still fine.
    assert _referenced_support_paths(
        skill + "see scripts/eval.ts:\n") == {"scripts/eval.ts"}
