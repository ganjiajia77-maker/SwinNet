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


def connect_bidirectional(c8, y, x, dy, dx, value=0.9):
    connect(c8, y, x, dy, dx, value)
    connect(c8, y + dy, x + dx, -dy, -dx, value)


def mark_horizontal(c8, y, xs, value=0.9):
    for x in xs:
        connect(c8, y, x, 0, 1, value)
        connect(c8, y, x, 0, -1, value)


def mark_vertical(c8, ys, x, value=0.9):
    for y in ys:
        connect(c8, y, x, 1, 0, value)
        connect(c8, y, x, -1, 0, value)


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


def test_intersection_preserves_multiple_branches():
    module = make_module()
    c8 = c8_grid(11, 11)
    mark_horizontal(c8, 5, range(2, 9))
    mark_vertical(c8, range(2, 9), 5)
    symmetric = module.build_symmetric_connectivity(c8)
    seeds = torch.zeros(1, 4, 11, 11)
    points = [(5, 2), (5, 8), (2, 5), (8, 5)]
    for index, (y, x) in enumerate(points):
        seeds[0, index, y, x] = 1
    reach = module.build_multisource_reachability(seeds, symmetric)
    assert reach[0, 0, 5, 8] > 0.5
    assert reach[0, 0, 2, 5] > 0.5
    assert reach[0, 0, 8, 5] > 0.5


def test_broken_middle_does_not_make_endpoint_pair_reachable():
    module = make_module()
    c8 = c8_grid(11, 11)
    mark_horizontal(c8, 5, range(2, 5))
    mark_horizontal(c8, 5, range(7, 10))
    symmetric = module.build_symmetric_connectivity(c8)
    seeds = torch.zeros(1, 2, 11, 11)
    seeds[0, 0, 5, 2] = 1
    seeds[0, 1, 5, 9] = 1
    reach = module.build_multisource_reachability(seeds, symmetric)
    assert reach[0, 0, 5, 9] < 0.01


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
