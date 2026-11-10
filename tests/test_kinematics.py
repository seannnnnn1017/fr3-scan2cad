"""FR3 nominal kinematics: FK sanity + IK success rate.  Run: python tests/test_kinematics.py (or pytest)."""
import numpy as np

import _path  # noqa: F401
import kinematics as kin


def test_fk_home_is_reasonable():
    q = np.array([0.0, -0.628, 0.0, -2.513, 0.0, 1.885, 0.785])
    T = kin.fk_link8(q)
    assert np.allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3), atol=1e-9)
    assert 0.2 < T[0, 3] < 0.6 and abs(T[1, 3]) < 1e-6 and 0.2 < T[2, 3] < 0.7


def ik_success_rate(n=100, seed=0):
    """Random reachable poses, seeded near the true solution, solved with ik_multi (what the planner calls)."""
    rng = np.random.default_rng(seed)
    ok = tot = 0
    while tot < n:
        q_true = kin.random_q(rng) * 0.5 + kin.MID * 0.5
        T = kin.fk_link8(q_true)
        if np.linalg.norm(T[:3, 3] - [0, 0, 0.333]) > 0.8 or T[2, 3] < 0.05:
            continue  # outside the planner's workspace; ik_multi rejects these by design
        tot += 1
        q = kin.ik_multi(T, [q_true + rng.normal(0, 0.15, 7)], rng=rng)
        if q is None:
            continue
        Tq = kin.fk_link8(q)
        assert np.linalg.norm(Tq[:3, 3] - T[:3, 3]) < 1e-3  # anything returned must be accurate
        assert np.linalg.norm(kin.rot_err(Tq[:3, :3], T[:3, :3])) < np.radians(0.5)
        ok += 1
    return ok / tot


def test_ik_success_rate():
    # KNOWN ISSUE (2026-11-10): ~12% of reachable poses stall at 1-4 mm, just over the 1 mm acceptance
    # (150 DLS iterations + null-space pull).  The planner then falls back to other views.
    # Raise this bar once the IK is improved.
    assert ik_success_rate() >= 0.8


if __name__ == "__main__":
    test_fk_home_is_reasonable()
    print(f"kinematics OK, IK success rate {ik_success_rate():.0%}")
