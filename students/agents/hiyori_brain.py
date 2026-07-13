from __future__ import annotations

import copy
import random
import string
from typing import Any, Sequence

import numpy as np


N_RAYS = 9
N_INPUTS = 18
N_HIDDEN = 16
N_ACTIONS = 4  # [throttle, brake, left, right]

# Fitness starts very low so a car that crashes before its first scoring
# update cannot outrank a car that makes genuine progress.
BASE_FITNESS = -100_000.0
DISTANCE_REWARD = 10_000.0
PACE_REWARD = 100.0
TIME_PENALTY = 5.0

# Fixed mixed mutation schedule. Each child independently receives one of
# four mutation strengths so the population always contains both refinements
# and exploratory variants, without relying on noisy lineage-level scores.
MUTATION_TIERS = (
    # probability, sigma, parameter coverage, neuron-reset chance
    (0.12, 0.020, 0.07, 0.01),  # fine tuning
    (0.48, 0.055, 0.18, 0.03),  # normal mutation
    (0.28, 0.120, 0.34, 0.07),  # strong exploration
    (0.12, 0.240, 0.55, 0.12),  # major exploration
)

# The engine activates controls only when an output is greater than 0.5.
# Throttle mutations are therefore moderated so useful children do not become
# permanently stationary from a small change near that hard threshold.
THROTTLE_MUTATION_SCALE = 0.45
THROTTLE_BIAS_MUTATION_SCALE = 0.25
MIN_MOVEMENT_THROTTLE = 0.53
MAX_THROTTLE_REPAIR_STEPS = 20
THROTTLE_REPAIR_STEP = 0.10

# Representative legal sensor patterns used only after mutation to confirm
# that the child can accelerate in at least one plausible driving situation.
MOVEMENT_TEST_PATTERNS = (
    (2, 4, 7, 10, 12, 10, 7, 4, 2),   # open corridor
    (1, 2, 4, 7, 10, 9, 7, 5, 3),     # mild right opening
    (3, 5, 7, 9, 10, 7, 4, 2, 1),     # mild left opening
    (2, 3, 5, 6, 7, 6, 5, 3, 2),      # moderate clearance
)



class hiyori_brain:
    """One-hidden-layer MLP racing agent for AI Cart V2.

    The network receives the nine engine raycasts plus features derived only
    from those raycasts and from ``passcardata``.  Its four sigmoid outputs
    map directly to throttle, brake, left, and right.
    """

    def __init__(self) -> None:
        self.score = BASE_FITNESS
        self.decider = 0
        self.x = 0.0
        self.y = 0.0
        self.speed = 0.0
        self.chars = string.ascii_letters + string.digits

        self.init_param()

    def init_param(self) -> None:
        # Xavier-style initialization keeps sigmoid activations in a useful
        # range while still providing diverse agents for evolutionary search.
        limit1 = np.sqrt(6.0 / (N_INPUTS + N_HIDDEN))
        limit2 = np.sqrt(6.0 / (N_HIDDEN + N_ACTIONS))
        self.W1 = np.random.uniform(-limit1, limit1, (N_HIDDEN, N_INPUTS))
        self.b1 = np.zeros(N_HIDDEN, dtype=float)
        self.W2 = np.random.uniform(-limit2, limit2, (N_ACTIONS, N_HIDDEN))
        self.b2 = np.array([0.85, -1.25, -0.20, -0.20], dtype=float)

        self.NAME = "Hiyori_MLP_" + "".join(random.choices(self.chars, k=5))
        self.store()

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        values = np.clip(values, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-values))

    def _prepare_inputs(self, data: Sequence[float]) -> np.ndarray:
        rays = np.asarray(data, dtype=float).ravel()
        if rays.size < N_RAYS:
            rays = np.pad(rays, (0, N_RAYS - rays.size))
        elif rays.size > N_RAYS:
            rays = rays[:N_RAYS]

        rays = np.nan_to_num(rays, nan=0.0, posinf=100.0, neginf=0.0)
        rays = np.maximum(rays, 0.0)

        # Normalize relative to the current visible range.  This avoids
        # depending on a hard-coded maximum ray distance or tile size.
        scale = max(float(np.max(rays)), 1.0)
        r = np.clip(rays / scale, 0.0, 1.0)

        # Ray order: [-90, -45, -20, -5, 0, 5, 20, 45, 90].
        left_open = float(np.mean(r[:4]))
        right_open = float(np.mean(r[5:]))
        near_left = float(np.mean(r[2:4]))
        near_right = float(np.mean(r[5:7]))
        front_open = float(np.mean(r[3:6]))
        edge_clearance = float(min(r[0], r[-1]))

        # Positive means more free track on the right; negative means left.
        steering_bias = right_open - left_open
        local_bias = near_right - near_left

        # Speed is normalized softly because the engine's documented maximum
        # is about 500, while clipping keeps this safe if that changes.
        speed_norm = float(np.clip(abs(self.speed) / 500.0, 0.0, 1.5) / 1.5)

        return np.concatenate(
            [
                r,
                np.array(
                    [
                        front_open,
                        left_open,
                        right_open,
                        steering_bias,
                        local_bias,
                        float(np.min(r[2:7])),
                        edge_clearance,
                        speed_norm,
                        1.0,  # constant feature lets evolution shift thresholds
                    ],
                    dtype=float,
                ),
            ]
        )

    def decide(self, data: Sequence[float]) -> np.ndarray:
        self.decider += 1
        x = self._prepare_inputs(data)
        hidden = np.tanh(self.W1 @ x + self.b1)
        outputs = self._sigmoid(self.W2 @ hidden + self.b2)

        # Prevent contradictory controls from being active together.  The
        # larger output survives; this remains deterministic and preserves
        # the network's learned preference.
        if outputs[0] > 0.5 and outputs[1] > 0.5:
            if outputs[0] >= outputs[1]:
                outputs[1] = 0.0
            else:
                outputs[0] = 0.0
        if outputs[2] > 0.5 and outputs[3] > 0.5:
            if outputs[2] >= outputs[3]:
                outputs[3] = 0.0
            else:
                outputs[2] = 0.0

        return outputs

    def _raw_outputs(self, data: Sequence[float]) -> np.ndarray:
        """Return network outputs without contradictory-action suppression."""
        x = self._prepare_inputs(data)
        hidden = np.tanh(self.W1 @ x + self.b1)
        return self._sigmoid(self.W2 @ hidden + self.b2)

    def _can_throttle(self) -> bool:
        """Check whether this brain can accelerate in a plausible situation."""
        old_speed = self.speed
        try:
            # Test at rest because that is where a newly spawned car begins.
            self.speed = 0.0
            return any(
                float(self._raw_outputs(pattern)[0]) >= MIN_MOVEMENT_THROTTLE
                for pattern in MOVEMENT_TEST_PATTERNS
            )
        finally:
            self.speed = old_speed

    def _repair_throttle_if_needed(self) -> None:
        """Gently lift throttle bias only when mutation made motion impossible.

        This does not force throttle in every situation. It only ensures that
        the child can exceed the engine's 0.5 action threshold for at least
        one representative open-road pattern.
        """
        for _ in range(MAX_THROTTLE_REPAIR_STEPS):
            if self._can_throttle():
                return
            self.b2[0] = float(
                np.clip(self.b2[0] + THROTTLE_REPAIR_STEP, -5.0, 5.0)
            )

        # Extremely unusual mutations can suppress throttle through the hidden
        # representation even after repeated bias repair. In that case, restore
        # a modest positive throttle bias and clear only the throttle output row.
        # Steering, braking, and hidden features remain mutated.
        if not self._can_throttle():
            self.W2[0] *= 0.5
            self.b2[0] = max(float(self.b2[0]), 0.85)

        # One final deterministic lift guarantees at least one legal movement
        # pattern without forcing throttle in every situation.
        while not self._can_throttle() and self.b2[0] < 5.0:
            self.b2[0] = min(5.0, float(self.b2[0]) + THROTTLE_REPAIR_STEP)

    def mutate(self) -> None:
        """Mutate a copied parent using a fixed mixture of strengths.

        The trainer gives every selected parent roughly the same number of
        children. This mutation mixture creates both refinements and exploratory
        jumps, while protecting the throttle pathway from mutations that would
        otherwise produce large numbers of permanently stationary children.
        """
        draw = float(np.random.random())
        cumulative = 0.0
        tier_index = len(MUTATION_TIERS) - 1
        sigma = 0.0
        probability = 0.0
        reset_probability = 0.0

        for index, (tier_chance, tier_sigma, coverage, reset_chance) in enumerate(MUTATION_TIERS):
            cumulative += tier_chance
            if draw < cumulative:
                tier_index = index
                sigma = tier_sigma
                probability = coverage
                reset_probability = reset_chance
                break

        changed = False

        # Hidden representation may mutate normally.
        for parameter in (self.W1, self.b1):
            mask = np.random.random(parameter.shape) < probability
            if np.any(mask):
                changed = True
            parameter += mask * np.random.normal(0.0, sigma, parameter.shape)

        # Steering and brake outputs mutate at full strength. The throttle row
        # mutates more gently because small changes near 0.5 can immobilize a car.
        output_noise = np.random.normal(0.0, sigma, self.W2.shape)
        output_noise[0] *= THROTTLE_MUTATION_SCALE
        output_mask = np.random.random(self.W2.shape) < probability
        if np.any(output_mask):
            changed = True
        self.W2 += output_mask * output_noise

        bias_noise = np.random.normal(0.0, sigma, self.b2.shape)
        bias_noise[0] *= THROTTLE_BIAS_MUTATION_SCALE
        bias_mask = np.random.random(self.b2.shape) < probability
        if np.any(bias_mask):
            changed = True
        self.b2 += bias_mask * bias_noise

        # Guarantee that every child differs from its parent, preferring a
        # steering/brake parameter rather than the fragile throttle pathway.
        if not changed:
            row = int(np.random.randint(1, N_ACTIONS))
            col = int(np.random.randint(0, N_HIDDEN))
            self.W2[row, col] += float(np.random.normal(0.0, sigma))

        # Stronger tiers occasionally reset one complete hidden feature.
        if np.random.random() < reset_probability:
            neuron = int(np.random.randint(0, N_HIDDEN))
            limit = np.sqrt(6.0 / (N_INPUTS + N_HIDDEN))
            self.W1[neuron] = np.random.uniform(-limit, limit, N_INPUTS)
            self.b1[neuron] = 0.0
            outgoing = np.random.normal(0.0, max(0.08, sigma), N_ACTIONS)
            outgoing[0] *= THROTTLE_MUTATION_SCALE
            self.W2[:, neuron] = outgoing

        # Prevent repeated large mutations from permanently saturating outputs.
        for parameter in (self.W1, self.b1, self.W2, self.b2):
            np.clip(parameter, -5.0, 5.0, out=parameter)

        # Keep exploratory mutations, but prevent genetically dead children
        # whose throttle never crosses the engine's action threshold.
        self._repair_throttle_if_needed()

        tier_names = ("FINE", "NORMAL", "STRONG", "MAJOR")
        self.NAME += (
            "_MUT_"
            + tier_names[tier_index]
            + "_"
            + "".join(random.choices(self.chars, k=3))
        )

        # A copied child must earn a fresh score in the new race.
        self.score = BASE_FITNESS
        self.decider = 0
        self.store()

    def calculate_score(self, distance: float, time: float, no: int) -> None:
        """
        Require a car to earn its way above an immediate-crash score.

        Every brain begins at ``BASE_FITNESS``. If a car crashes before the
        engine calls this method, its score therefore remains very low.
        Stationary cars become slightly worse over time, while any genuine
        distance earns a large positive reward. Pace breaks ties in favour of
        faster progress. ``no`` is ignored because its meaning is not stable.
        """
        safe_distance = max(float(distance), 0.0)
        safe_time = max(float(time), 0.05)
        pace = safe_distance / safe_time

        self.score = (
            BASE_FITNESS
            + safe_distance * DISTANCE_REWARD
            + pace * PACE_REWARD
            - safe_time * TIME_PENALTY
        )

    def passcardata(self, x: float, y: float, speed: float) -> None:
        self.x = float(x)
        self.y = float(y)
        self.speed = float(speed)

    def getscore(self) -> float:
        return float(self.score)

    def store(self) -> None:
        self.parameters = copy.deepcopy(
            {
                "W1": self.W1,
                "b1": self.b1,
                "W2": self.W2,
                "b2": self.b2,
                "NAME": self.NAME,
                "architecture": np.array([N_INPUTS, N_HIDDEN, N_ACTIONS]),
            }
        )

    def get_parameters(self) -> dict[str, Any]:
        self.store()
        return copy.deepcopy(self.parameters)

    def set_parameters(self, parameters: Any) -> None:
        if isinstance(parameters, np.lib.npyio.NpzFile):
            params = {key: parameters[key] for key in parameters.files}
        else:
            params = copy.deepcopy(parameters)

        self.W1 = np.asarray(params["W1"], dtype=float).copy()
        self.b1 = np.asarray(params["b1"], dtype=float).copy()
        self.W2 = np.asarray(params["W2"], dtype=float).copy()
        self.b2 = np.asarray(params["b2"], dtype=float).copy()
        self.NAME = str(params["NAME"])

        expected = (
            (N_HIDDEN, N_INPUTS),
            (N_HIDDEN,),
            (N_ACTIONS, N_HIDDEN),
            (N_ACTIONS,),
        )
        actual = (self.W1.shape, self.b1.shape, self.W2.shape, self.b2.shape)
        if actual != expected:
            raise ValueError(
                "Save does not match hiyori_brain MLP architecture: "
                f"expected {expected}, got {actual}"
            )

        # Runtime fitness is never restored from the save. A freshly loaded
        # car must earn progress just like a newly created one.
        self.score = BASE_FITNESS
        self.decider = 0
        self.store()


if __name__ == "__main__":
    brain = hiyori_brain()
    brain.passcardata(0.0, 0.0, 120.0)
    print(brain.decide([2, 4, 7, 10, 12, 10, 7, 4, 2]))
