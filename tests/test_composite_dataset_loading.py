from reap.data import parse_composite_dataset_spec


REAL_COMPOSITE_SPEC = (
    "theblackcat102/evol-codealpaca-v1:8,"
    "Salesforce/xlam-function-calling-60k:8,"
    "open-r1/Mixture-of-Thoughts[code]:8,"
    "open-r1/Mixture-of-Thoughts[math]:8,"
    "open-r1/Mixture-of-Thoughts[science]:8,"
    "SWE-bench/SWE-smith-trajectories(tool):8"
)


def test_parse_composite_dataset_spec():
    composite_components = parse_composite_dataset_spec(
        REAL_COMPOSITE_SPEC,
        default_split="train",
    )
    assert composite_components is not None
    assert [
        (component.name, component.subset, component.split, component.num_batches)
        for component in composite_components
    ] == [
        ("theblackcat102/evol-codealpaca-v1", None, "train", 8),
        ("Salesforce/xlam-function-calling-60k", None, "train", 8),
        ("open-r1/Mixture-of-Thoughts", "code", "train", 8),
        ("open-r1/Mixture-of-Thoughts", "math", "train", 8),
        ("open-r1/Mixture-of-Thoughts", "science", "train", 8),
        ("SWE-bench/SWE-smith-trajectories", None, "tool", 8),
    ]
