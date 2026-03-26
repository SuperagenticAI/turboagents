from turboagents.bench.datasets import DATASETS, get_dataset, make_vector_dataset


def test_dataset_registry_contains_expected_profiles() -> None:
    assert "tiny-kv" in DATASETS
    assert "tiny-rag" in DATASETS
    assert get_dataset("paper-sim").dim == 128


def test_make_vector_dataset_is_deterministic() -> None:
    base_a, queries_a = make_vector_dataset("tiny-rag")
    base_b, queries_b = make_vector_dataset("tiny-rag")
    assert base_a.shape == (256, 128)
    assert queries_a.shape == (16, 128)
    assert (base_a == base_b).all()
    assert (queries_a == queries_b).all()
