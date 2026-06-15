from vdm_spike.power import is_underpowered, min_detectable_effect


def test_mde_decreases_with_sample_size():
    small = min_detectable_effect(n=50)
    large = min_detectable_effect(n=5000)
    assert small > large > 0


def test_underpowered_flag():
    assert is_underpowered(n=20, target_effect=0.1) is True
    assert is_underpowered(n=5000, target_effect=0.5) is False
