# pylint: disable=line-too-long, missing-function-docstring, missing-module-docstring
import pytest
from src.utils.matrix import (
    load_parameters,
    load_principles,
    get_inventive_principles,
    get_random_principles,
    get_principle_description,
    get_principle_name
)

def test_load_parameters():
    params = load_parameters()
    assert isinstance(params, dict)
    assert len(params['data']) > 0

def test_load_principles():
    principles = load_principles()
    assert isinstance(principles, dict)
    assert len(principles['data']) > 0

def test_get_inventive_principles():
    empty = get_inventive_principles(1, 2)
    assert isinstance(empty, list)
    assert len(empty) == 0
    principles = get_inventive_principles(1, 5)
    assert isinstance(principles, list)
    assert all(isinstance(p, int) for p in principles)
    assert 1 <= min(principles) <= max(principles) <= 40
    assert set(principles) == {17, 29, 34, 38}

def test_get_random_principles():
    principles = get_random_principles()
    assert isinstance(principles, list)
    assert len(principles) == 4
    assert all(isinstance(p, int) for p in principles)
    assert 1 <= min(principles) <= max(principles) <= 40

def test_get_principle_description():
    description = get_principle_description(1)
    assert isinstance(description, str)
    assert len(description) > 0

def test_get_principle_name():
    name = get_principle_name(1)
    assert isinstance(name, str)
    assert len(name) > 0
    assert name == "Segmentation"

@pytest.mark.parametrize("invalid_index", [-1, 0, 41])
def test_invalid_principle_index(invalid_index):
    with pytest.raises(IndexError):
        get_principle_description(invalid_index)
    with pytest.raises(IndexError):
        get_principle_name(invalid_index)

if __name__ == "__main__":
    pytest.main([__file__])
