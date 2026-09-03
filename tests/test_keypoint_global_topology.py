import torch

from networks.keypoint_global_topology import KeypointGuidedGlobalTopology


def make_module():
    return KeypointGuidedGlobalTopology(
        channels=8,
        max_nodes=8,
        heads=2,
        reach_hops=8,
        skeleton_threshold=0.1,
        connectivity_threshold=0.2,
        enabled=True,
    )


def c8_grid(height, width):
    return torch.full((1, 8, height, width), 0.01)


def connect(c8, y, x, dy, dx, value=0.9):
    directions = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))
    index = directions.index((dy, dx))
    c8[0, index, y, x] = value


def test_straight_reachability():
    module = make_module()
    c8 = c8_grid(9, 9)
    for x in range(2, 7):
        connect(c8, 4, x, 0, 1)
        connect(c8, 4, x, 0, -1)
    symmetric = module.build_symmetric_connectivity(c8)
    seeds = torch.zeros(1, 2, 9, 9)
    seeds[0, 0, 4, 2] = 1
    seeds[0, 1, 4, 6] = 1
    reach = module.build_multisource_reachability(seeds, symmetric)
    assert reach[0, 0, 4, 6] > 0.5


def test_bent_path_reachability_without_line_rasterization():
    module = make_module()
    c8 = c8_grid(9, 9)
    for x in range(2, 6):
        connect(c8, 4, x, 0, 1)
        connect(c8, 4, x, 0, -1)
    for y in range(4, 7):
        connect(c8, y, 5, 1, 0)
        connect(c8, y, 5, -1, 0)
    symmetric = module.build_symmetric_connectivity(c8)
    seeds = torch.zeros(1, 2, 9, 9)
    seeds[0, 0, 4, 2] = 1
    seeds[0, 1, 6, 5] = 1
    reach = module.build_multisource_reachability(seeds, symmetric)
    assert reach[0, 0, 6, 5] > 0.5


def test_parallel_paths_are_not_connected():
    module = make_module()
    c8 = c8_grid(9, 9)
    for x in range(2, 7):
        for y in (3, 5):
            connect(c8, y, x, 0, 1)
            connect(c8, y, x, 0, -1)
    symmetric = module.build_symmetric_connectivity(c8)
    seeds = torch.zeros(1, 2, 9, 9)
    seeds[0, 0, 3, 4] = 1
    seeds[0, 1, 5, 4] = 1
    reach = module.build_multisource_reachability(seeds, symmetric)
    assert reach[0, 0, 5, 4] < 0.01


def test_disabled_and_zero_alpha_are_identity():
    feature = torch.randn(1, 8, 9, 9)
    skeleton = torch.ones(1, 1, 9, 9)
    connectivity = torch.rand(1, 8, 9, 9)
    direction = torch.randn(1, 2, 9, 9)
    module = make_module()
    module.enable_global_topology = False
    assert torch.equal(module(feature, skeleton, connectivity, direction), feature)
    module.enable_global_topology = True
    module.raw_alpha.data.zero_()
    assert torch.equal(module(feature, skeleton, connectivity, direction), feature)
