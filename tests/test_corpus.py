from vdm_spike.corpus import BALANCED_MIX, SHIFTED_MIX, make_docs


def test_make_docs_is_deterministic_for_same_seed():
    a = make_docs(seed=0, n=50, topic_mix=BALANCED_MIX)
    b = make_docs(seed=0, n=50, topic_mix=BALANCED_MIX)
    assert [d.text for d in a] == [d.text for d in b]


def test_make_docs_count_and_unique_ids():
    docs = make_docs(seed=1, n=120, topic_mix=BALANCED_MIX)
    assert len(docs) == 120
    assert len({d.id for d in docs}) == 120


def test_shifted_mix_changes_topic_distribution():
    base = make_docs(seed=2, n=200, topic_mix=BALANCED_MIX)
    shifted = make_docs(seed=2, n=200, topic_mix=SHIFTED_MIX)
    base_topics = [d.topic for d in base]
    shifted_topics = [d.topic for d in shifted]
    assert base_topics != shifted_topics
