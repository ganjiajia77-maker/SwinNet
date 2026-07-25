import torch
import torch.nn as nn
import torch.nn.functional as F

GRAPH_DIFFUSION_DIRECTIONS = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
)


def _shift_feature_map(x, dy, dx):
    _, _, height, width = x.shape
    pad_left = max(dx, 0)
    pad_right = max(-dx, 0)
    pad_top = max(dy, 0)
    pad_bottom = max(-dy, 0)
    padded = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
    y0 = max(-dy, 0)
    x0 = max(-dx, 0)
    return padded[:, :, y0:y0 + height, x0:x0 + width]


class SimpleConnectivityDiffusion(nn.Module):
    """F' = F + gamma * mean_k(C_k * shift(F)). Uses predicted connectivity only."""

    def __init__(self, gamma_init=0.05):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def connectivity_message(self, feature, connectivity_prob):
        message = torch.zeros_like(feature)
        for channel_idx, (dy, dx) in enumerate(GRAPH_DIFFUSION_DIRECTIONS):
            shifted = _shift_feature_map(feature, dy, dx)
            message = message + connectivity_prob[:, channel_idx:channel_idx + 1] * shifted
        return message / float(len(GRAPH_DIFFUSION_DIRECTIONS))

    def diffuse(self, feature, connectivity_prob):
        message = self.connectivity_message(feature, connectivity_prob)
        return feature + self.gamma * message, message

    def forward(self, feature, connectivity_prob):
        return self.diffuse(feature, connectivity_prob)[0]


class SkeletonConnectivityGraphDiffusion(nn.Module):
    """Polynomial symmetric graph diffusion before structure gate.

    A_ij = C_ij * (1 + alpha * S_i * S_j)
    T = D^{-1/2} A D^{-1/2}
    F_d = sum_{k=0..K} theta_k * T^k X
    F' = X + gamma * F_d
    """

    def __init__(self, alpha=1.0, gamma_init=0.05, poly_order=3):
        super().__init__()
        self.alpha = float(alpha)
        self.poly_order = int(poly_order)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        thetas = torch.zeros(self.poly_order + 1)
        if self.poly_order >= 1:
            thetas[1] = 1.0
        self.thetas = nn.Parameter(thetas)

    def build_adjacency(self, skeleton_prob, connectivity_prob):
        adjacency_weights = []
        for channel_idx, (dy, dx) in enumerate(GRAPH_DIFFUSION_DIRECTIONS):
            neighbor_skeleton = _shift_feature_map(skeleton_prob, dy, dx)
            connectivity_ij = connectivity_prob[:, channel_idx:channel_idx + 1]
            skeleton_factor = 1.0 + self.alpha * skeleton_prob * neighbor_skeleton
            adjacency_weights.append(connectivity_ij * skeleton_factor)
        return adjacency_weights

    @staticmethod
    def node_degree(adjacency_weights):
        return sum(adjacency_weights)

    def apply_normalized_adjacency(self, feature, adjacency_weights, degree):
        message = torch.zeros_like(feature)
        for weight, (dy, dx) in zip(adjacency_weights, GRAPH_DIFFUSION_DIRECTIONS):
            neighbor_degree = _shift_feature_map(degree, dy, dx)
            norm = (degree * neighbor_degree).clamp_min(1e-6).sqrt()
            t_weight = weight / norm
            message = message + t_weight * _shift_feature_map(feature, dy, dx)
        return message

    def polynomial_diffusion(self, feature, adjacency_weights, degree):
        t_power = feature
        f_diff = self.thetas[0] * feature
        for order in range(1, self.poly_order + 1):
            t_power = self.apply_normalized_adjacency(
                t_power,
                adjacency_weights,
                degree,
            )
            f_diff = f_diff + self.thetas[order] * t_power
        return f_diff

    def diffuse(self, feature, skeleton_prob, connectivity_prob):
        adjacency_weights = self.build_adjacency(skeleton_prob, connectivity_prob)
        degree = self.node_degree(adjacency_weights)
        f_diff = self.polynomial_diffusion(feature, adjacency_weights, degree)
        return feature + self.gamma * f_diff, f_diff

    def forward(self, feature, skeleton_prob, connectivity_prob):
        return self.diffuse(feature, skeleton_prob, connectivity_prob)[0]


class DirectionAwareGraphDiffusion(nn.Module):
    """Soft grid-graph message passing over predicted skeleton/connectivity/direction."""

    def __init__(
        self,
        channels,
        alpha=1.0,
        beta=1.0,
        gamma_init=0.05,
        use_message_mlp=False,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        self.use_message_mlp = bool(use_message_mlp)
        if self.use_message_mlp:
            self.message_mlp = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.message_mlp = None

    @staticmethod
    def _shift(x, dy, dx):
        return _shift_feature_map(x, dy, dx)

    @staticmethod
    def direction_similarity(direction_i, direction_j):
        direction_i = F.normalize(direction_i, dim=1, eps=1e-6)
        direction_j = F.normalize(direction_j, dim=1, eps=1e-6)
        cosine = (direction_i * direction_j).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
        return (cosine + 1.0) * 0.5

    def build_adjacency(self, skeleton_prob, connectivity_prob, direction_field):
        """A_ij = C_ij * (1 + alpha*S_i*S_j) * (1 + beta*A_direction(i,j))."""
        adjacency_weights = []
        for channel_idx, (dy, dx) in enumerate(GRAPH_DIFFUSION_DIRECTIONS):
            neighbor_skeleton = self._shift(skeleton_prob, dy, dx)
            connectivity_ij = connectivity_prob[:, channel_idx:channel_idx + 1]
            skeleton_factor = 1.0 + self.alpha * skeleton_prob * neighbor_skeleton
            neighbor_direction = self._shift(direction_field, dy, dx)
            direction_alignment = self.direction_similarity(
                direction_field,
                neighbor_direction,
            )
            direction_factor = 1.0 + self.beta * direction_alignment
            adjacency_weights.append(
                connectivity_ij * skeleton_factor * direction_factor
            )
        return adjacency_weights

    @staticmethod
    def row_normalize_adjacency(adjacency_weights):
        row_sum = sum(adjacency_weights)
        return [
            weight / row_sum.clamp_min(1e-6)
            for weight in adjacency_weights
        ]

    def diffuse(self, feature, skeleton_prob, connectivity_prob, direction_field):
        adjacency_weights = self.build_adjacency(
            skeleton_prob,
            connectivity_prob,
            direction_field,
        )
        normalized_weights = self.row_normalize_adjacency(adjacency_weights)
        message = torch.zeros_like(feature)
        for weight, (dy, dx) in zip(normalized_weights, GRAPH_DIFFUSION_DIRECTIONS):
            neighbor_feature = self._shift(feature, dy, dx)
            message = message + weight * neighbor_feature
        if self.message_mlp is not None:
            message = self.message_mlp(message)
        return feature + self.gamma * message, message

    def forward(self, feature, skeleton_prob, connectivity_prob, direction_field):
        return self.diffuse(
            feature,
            skeleton_prob,
            connectivity_prob,
            direction_field,
        )[0]
