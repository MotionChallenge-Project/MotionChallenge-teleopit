"""TeleopIT's released VelCmdHistory controller, using the shared state API."""
from collections import deque
import numpy as np
from motion_challenge.adapters.onnx import ONNXController


def quat(value):
    return np.asarray(value, np.float32)[[3, 0, 1, 2]]


def inv(q):
    q = q.copy(); q[1:] *= -1
    return q


def mul(a, b):
    w, x, y, z = a; v, i, j, k = b
    return np.asarray([w*v-x*i-y*j-z*k, w*i+x*v+y*k-z*j,
                       w*j-x*k+y*v+z*i, w*k+x*j-y*i+z*v], np.float32)


def rotate(q, v):
    return mul(mul(q, np.r_[np.float32(0), v]), inv(q))[1:]


def rotation6(q):
    w, x, y, z = q
    return np.asarray([1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*y+w*z),
                       1-2*(x*x+z*z), 2*(x*z-w*y), 2*(y*z+w*x)], np.float32)


class TeleopIT(ONNXController):
    action_output = 'actions'
    raw_action_history = True

    def __init__(self, root, runtime):
        super().__init__(root, runtime)
        self.default = self.default.astype(np.float32)
        self.scale = self.scale.astype(np.float32)

    def reset(self, state, reference, seed):
        super().reset(state, reference, seed)
        self.previous = None
        self.history = deque(maxlen=10)

    def observations(self, state, reference):
        ref = reference[0]
        actual = {b.name:b for b in state.bodies}['torso_link']
        desired = {b.name:b for b in ref.bodies}['torso_link']
        aq, rq, baseq = quat(actual.orientation_xyzw), quat(desired.orientation_xyzw), quat(state.root_orientation_xyzw)
        q = np.asarray(state.joint_position_rad, np.float32)[self.order]
        qd = np.asarray(state.joint_velocity_rad_s, np.float32)[self.order]
        tq = np.asarray(ref.joint_position_rad, np.float32)[self.order]
        rp = np.asarray(desired.position_world_m, np.float32)
        dt = np.float32(self.runtime['control_period_s'])
        if self.previous is None:
            tv, lv, av = np.zeros(29, np.float32), np.zeros(3, np.float32), np.zeros(3, np.float32)
        else:
            pq, pp, pr = self.previous
            tv = (tq-pq)*np.float32(50)
            lv = (rp-pp)/dt
            delta = mul(rq, inv(pr))
            if delta[0] < 0: delta = -delta
            half = np.float32(np.arccos(float(np.clip(delta[0], -1., 1.))))
            sine = np.float32(np.sin(half))
            av = delta[1:]/sine*2.*half/dt if sine > 1e-6 else np.zeros(3, np.float32)
        self.previous = (tq.copy(), rp.copy(), rq.copy())
        gravity = np.asarray([0., 0., -1.], np.float32)
        w = rotate(inv(baseq), np.asarray(state.root_angular_velocity_world_rad_s, np.float32))
        obs = np.concatenate([tq, tv, rotation6(mul(inv(aq), rq)), w, q-self.default,
            qd, self.last, rotate(inv(baseq), gravity), rotate(inv(aq), lv),
            rotate(inv(aq), av), rotate(inv(rq), gravity), rp[2:3]]).astype(np.float32)
        if not self.history:
            self.history.extend(obs.copy() for _ in range(10))
        else:
            self.history.append(obs.copy())
        return {'obs':obs[None], 'obs_history':np.stack(self.history)[None]}


def create(root, runtime):
    return TeleopIT(root, runtime)
