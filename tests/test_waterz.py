from math import isclose

import numpy as np

import waterz as wz


def test_evaluate() -> None:
    np.random.seed(0)
    seg = np.random.randint(500, size=(3, 3, 3), dtype=np.uint64)
    scores = wz.evaluate(seg, seg)
    assert scores["voi_split"] == 0.0
    assert scores["voi_merge"] == 0.0
    assert scores["rand_split"] == 1.0
    assert scores["rand_merge"] == 1.0

    seg2 = np.random.randint(500, size=(3, 3, 3), dtype=np.uint64)
    scores = wz.evaluate(seg, seg2)

    # print('scores: ', scores)
    # Note that these values are from the first run
    # I have not double checked that this is correct or not.
    # This assertion only make sure that future changes of
    # code will not change the result of the evaluation
    assert isclose(scores["rand_split"], 0.8181818181818182)
    assert isclose(scores["rand_merge"], 0.8709677419354839)
    assert isclose(scores["voi_split"], 0.22222222222222232)
    assert isclose(scores["voi_merge"], 0.14814814814814792)


def test_agglomerate() -> None:
    np.random.seed(0)
    # affinities is a [3,depth,height,width] numpy array of float32
    affinities = np.random.rand(3, 4, 4, 4).astype(np.float32)

    thresholds = [0, 100, 200]
    results = list(wz.agglomerate(affinities, thresholds))
    assert len(results) == 3
    for segmentation in results:
        assert isinstance(segmentation, np.ndarray)
        assert segmentation.shape == (4, 4, 4)
        assert segmentation.dtype == np.uint64
        # just what I observed... from my random test
        # change when better test data is available
        assert np.all(segmentation == 1)


def test_waterz_can_return_region_graph() -> None:
    affinities = np.ones((3, 2, 2, 2), dtype=np.float32)
    fragments = np.zeros((2, 2, 2), dtype=np.uint64)
    fragments[:, :, 0] = 1
    fragments[:, :, 1] = 2

    expected = []
    for segmentation, region_graph in wz.agglomerate(
        affinities,
        [0.0],
        fragments=fragments,
        return_region_graph=True,
    ):
        expected.append((np.array(segmentation, copy=True), region_graph))

    actual = wz.waterz(
        affinities,
        [0.0],
        fragments=fragments,
        return_region_graph=True,
    )

    assert len(actual) == len(expected) == 1
    expected_segmentation, expected_region_graph = expected[0]
    actual_segmentation, actual_region_graph = actual[0]
    np.testing.assert_array_equal(actual_segmentation, expected_segmentation)
    assert actual_region_graph == expected_region_graph


def test_waterz_as_dict_preserves_tuple_results() -> None:
    affinities = np.ones((3, 2, 2, 2), dtype=np.float32)
    fragments = np.zeros((2, 2, 2), dtype=np.uint64)
    fragments[:, :, 0] = 1
    fragments[:, :, 1] = 2

    actual = wz.waterz(
        affinities,
        [0.0],
        fragments=fragments,
        return_region_graph=True,
        as_dict=True,
    )

    assert list(actual.keys()) == [0.0]
    segmentation, region_graph = actual[0.0]
    assert isinstance(segmentation, np.ndarray)
    assert segmentation.shape == (2, 2, 2)
    assert isinstance(region_graph, list)
    assert region_graph


def test_merge_function_to_scoring_accepts_affmean_alias() -> None:
    assert wz.merge_function_to_scoring("affmean") == (
        "OneMinus<MeanAffinity<RegionGraphType, ScoreValue>>"
    )
