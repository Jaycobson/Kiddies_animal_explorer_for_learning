from pipeline.script_parser import Scene, build_prompt, parse_characters, parse_script


def test_splits_on_scene_markers():
    s = parse_script("Scene 1: Amina walks.\n\nScene 2: Amina rests.")
    assert [x.text for x in s] == ["Amina walks.", "Amina rests."]


def test_splits_on_blank_lines():
    assert len(parse_script("One.\n\nTwo.\n\nThree.")) == 3


def test_prompt_only_includes_present_characters():
    chars = parse_characters("Amina: green headwrap\nNgozi: blue scrubs")
    p = build_prompt(Scene(1, "Amina smiles."), chars, "flat cartoon")
    assert "green headwrap" in p and "blue scrubs" not in p and "flat cartoon" in p
