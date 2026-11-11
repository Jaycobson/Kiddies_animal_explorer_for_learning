import pytest

from pipeline.planner import MAX_SCENES, normalize_plan


def test_caps_scenes_and_cleans_characters():
    data = {"title": "T", "visual_style": "storybook",
            "characters": [{"name": "Ife", "description": "orange dress"}, {"name": "", "description": "x"}],
            "scenes": [{"narration": f"Scene {i}", "visual": "v"} for i in range(12)]}
    plan = normalize_plan(data)
    assert len(plan["scenes"]) == MAX_SCENES and plan["characters"] == {"Ife": "orange dress"}


def test_empty_plan_raises():
    with pytest.raises(ValueError):
        normalize_plan({"scenes": []})
