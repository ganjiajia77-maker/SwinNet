"""Source selection for V9. Local mode delegates to the unmodified V9 forward."""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from networks.reliable_directional_repair import DirectionalDynamics, DIRECTIONS, shifted

RADII = (1, 2, 4, 8)
SOURCES = tuple((dy, dx, radius) for dy, dx in DIRECTIONS for radius in RADII)


def sample_offset(value, dy, dx):
    """Read p+(dy,dx); x is width, y is height; zero outside pixel centers."""
    h, w = value.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(h, device=value.device, dtype=value.dtype),
                            torch.arange(w, device=value.device, dtype=value.dtype), indexing='ij')
    grid = torch.stack((2 * (xx + dx + .5) / w - 1,
                        2 * (yy + dy + .5) / h - 1), -1)
    return F.grid_sample(value, grid[None].expand(value.shape[0], -1, -1, -1),
                         mode='bilinear', padding_mode='zeros', align_corners=False)


def local_target(state, support, weights):
    mass = torch.zeros_like(state)
    numerator = torch.zeros_like(state)
    for index, (dy, dx) in enumerate(DIRECTIONS):
        source = weights[:, index:index+1] * shifted(support, dy, dx)
        mass = mass + source
        numerator = numerator + source * shifted(state, dy, dx)
    target = torch.where(mass > 1e-6, numerator / mass.clamp_min(1e-6), state)
    return target, mass.clamp(0., 1.), mass


class SourceSelector(nn.Module):
    """One shared scorer for 32 directed relations, plus an explicit stay logit.

    Corridor P0/Q and visual samples are evidence, never hard validity gates.
    R's evolving support follows V9; only source coordinates are hard-masked.
    Checkpointing recomputes the wide per-relation input on backward, avoiding
    retaining 32 large concatenations per step on a 128x128 work grid.
    """
    def __init__(self, width=32, embedding=8, hidden=32, recompute=True):
        super().__init__()
        self.project = nn.Conv2d(width, embedding, 1)
        channels = 4 * embedding + 9 * (embedding + 3) + 11
        self.score = nn.Sequential(nn.Conv2d(channels, hidden, 1), nn.GELU(),
                                   nn.Conv2d(hidden, 16, 1), nn.GELU(), nn.Conv2d(16, 1, 1))
        self.stay = nn.Sequential(nn.Conv2d(embedding + 4, 16, 1), nn.GELU(), nn.Conv2d(16, 1, 1))
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)
        nn.init.zeros_(self.stay[-1].weight)
        nn.init.constant_(self.stay[-1].bias, 2.)
        self.recompute = recompute

    def _score(self, feature, state, support, p0, q, direction_weight, dy, dx, radius):
        source = shifted(feature, dy * radius, dx * radius)
        corridor = torch.cat((feature, p0, q, torch.ones_like(p0)), 1)
        samples = []
        norm = math.hypot(dy, dx)
        for fraction in (.25, .5, .75):
            for lateral in (-.5, 0., .5):
                samples.append(sample_offset(corridor, dy * radius * fraction + dx / norm * lateral,
                                              dx * radius * fraction - dy / norm * lateral))
        geometry = torch.cat((state, shifted(state, dy*radius, dx*radius),
                              shifted(support, dy*radius, dx*radius), direction_weight,
                              p0, shifted(p0, dy*radius, dx*radius), q,
                              shifted(q, dy*radius, dx*radius),
                              torch.full_like(state, radius / 8.),
                              torch.full_like(state, dy / norm), torch.full_like(state, dx / norm)), 1)
        return self.score(torch.cat((feature, source, feature-source, feature*source, *samples, geometry), 1))

    def forward(self, visual, state, support, p0, q, weights):
        feature = self.project(visual)
        logits, values, validities = [], [], []
        ones = torch.ones_like(state)
        for index, (dy, dx, radius) in enumerate(SOURCES):
            arguments = (feature, state, support, p0, q, weights[:, index//4:index//4+1], dy, dx, radius)
            score = (checkpoint(self._score, *arguments, use_reentrant=False)
                     if self.recompute and torch.is_grad_enabled() else self._score(*arguments))
            prior = shifted(support, dy*radius, dx*radius)
            valid = shifted(ones, dy*radius, dx*radius) > .5
            logits.append((score + torch.log(prior.clamp_min(1e-6))).masked_fill(~valid, -torch.inf))
            values.append(shifted(state, dy*radius, dx*radius))
            validities.append(valid)
        stay_logit = self.stay(torch.cat((feature, state, p0, q, support), 1))
        probabilities = torch.cat((stay_logit, *logits), 1).softmax(1)
        source_values = torch.cat(values, 1)
        stay_weight, source_weights = probabilities[:, :1], probabilities[:, 1:]
        target = stay_weight * state + (source_weights * source_values).sum(1, keepdim=True)
        valid = torch.cat(validities, 1)
        # Explicit fallback also protects against floating roundoff in the all-invalid case.
        target = torch.where(valid.any(1, keepdim=True), target, state)
        return target, 1. - stay_weight, {
            'selector_weights': probabilities, 'selector_source_values': source_values,
            'selector_valid': valid, 'selector_stay': stay_weight,
        }


class SelectedDynamics(DirectionalDynamics):
    def __init__(self, original, mode, width=32, seed=1234, recompute=True):
        super().__init__(width, steps=original.steps)
        if mode not in ('local', 'multiscale'):
            raise ValueError('transport mode must be local or multiscale')
        self.rates = original.rates
        self.mode = mode
        self.blend = 1.
        # Creating a selector must not change the paired runs' RNG sequences.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.selector = SourceSelector(width, recompute=recompute) if mode == 'multiscale' else None

    def forward(self, visual, p0, q, reliable, proposal, accept, weights,
                capture_diagnostics=False, transport_gate=None,
                freeze_generated_sources=False, disable_transport=False,
                disable_proposal=False, state_gate=None, base_gate_logit=None):
        if self.mode == 'local':
            return super().forward(visual, p0, q, reliable, proposal, accept, weights,
                capture_diagnostics, transport_gate, freeze_generated_sources,
                disable_transport, disable_proposal, state_gate, base_gate_logit)
        if not 0. <= self.blend <= 1.:
            raise ValueError('The fixed schedule blend must lie in [0,1]')
        state = p0
        states, masses, availabilities = [state], [], []
        keys = ('alpha', 'beta', 'gamma', 'stay', 'm', 'transport_delta', 'proposal_delta',
                'anchor_delta', 'g_step', 'state_gate_delta', 'transport_rate',
                'selector_weights', 'selector_source_values', 'selector_stay',
                'selector_target', 'selector_available')
        details = {key: [] for key in keys}
        for _ in range(self.steps):
            if freeze_generated_sources:
                support = reliable
            elif transport_gate is not None:
                gained = ((state-p0).clamp_min(0.) / (1.-p0).clamp_min(1e-5)).clamp_max(1.)
                support = reliable + (1.-reliable)*gained
            else:
                support = reliable + (1.-reliable)*accept*state
            multi, multi_available, selected = self.selector(visual, state, support, p0, q, weights)
            if self.blend < 1.:
                local, local_available, local_mass = local_target(state, support, weights)
                propagated = local if self.blend == 0. else (1.-self.blend)*local + self.blend*multi
                available = local_available if self.blend == 0. else (1.-self.blend)*local_available + self.blend*multi_available
                mass = local_mass if self.blend == 0. else available
            else:
                propagated, available, mass = multi, multi_available, multi_available
            coefficients = self.rates(torch.cat((visual, p0, q, reliable, proposal, accept, state, available), 1)).softmax(1)
            state_delta = torch.zeros_like(state)
            step_gate = transport_gate
            if state_gate is not None:
                if base_gate_logit is None or transport_gate is None:
                    raise ValueError('State gate requires the original V9 gate')
                state_delta = state_gate(torch.cat((state, propagated-state, available,
                    coefficients[:, 1:2], proposal-state, p0-state), 1))
                step_gate = torch.sigmoid(base_gate_logit + state_delta)
            transport = (coefficients[:, 1:2]*step_gate if transport_gate is not None
                         else accept*coefficients[:, 1:2]*available)
            propose = .2*accept*coefficients[:, 2:3]
            if disable_transport:
                transport = torch.zeros_like(transport)
            if disable_proposal:
                propose = torch.zeros_like(propose)
            anchor = coefficients[:, 3:4]
            stay = 1.-transport-propose-anchor
            if capture_diagnostics:
                values = dict(alpha=transport, beta=propose, gamma=anchor, stay=stay, m=propagated,
                    transport_delta=transport*(propagated-state), proposal_delta=propose*(proposal-state),
                    anchor_delta=anchor*(p0-state), g_step=step_gate if step_gate is not None else torch.zeros_like(state),
                    state_gate_delta=state_delta, transport_rate=coefficients[:, 1:2],
                    selector_target=multi, selector_available=multi_available)
                values.update({key: selected[key] for key in keys if key in selected})
                for key in keys:
                    details[key].append(values[key])
            state = stay*state + transport*propagated + propose*proposal + anchor*p0
            states.append(state); masses.append(mass); availabilities.append(available)
        result = (states, masses, availabilities)
        return (*result, {key: torch.stack(value, 1) for key, value in details.items()}) if capture_diagnostics else result


def install(model, mode, seed=1234, recompute=True):
    original = model.dynamics
    # Even the temporary parent rates constructor is isolated from the run RNG.
    with torch.random.fork_rng(devices=[]):
        replacement = SelectedDynamics(original, mode, model.width, seed, recompute)
    model.dynamics = replacement.to(next(model.parameters()).device)
    return model


def transition(epoch, warmup_epochs=3):
    if warmup_epochs < 0 or epoch < 1:
        raise ValueError('Invalid transition schedule')
    return min(1., (epoch-1)/warmup_epochs) if warmup_epochs else 1.
