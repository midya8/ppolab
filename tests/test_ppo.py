import numpy as np
import pytest

from ppolab.nets import ActorCritic
from ppolab.ppo import ppo_loss_and_grads, log_softmax


def tiny_logits_values(seed=0, B=6, hidden=5):
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(B, 2))
    values = rng.normal(size=B)
    actions = rng.integers(0, 2, size=B)
    old_logp = log_softmax(logits)[np.arange(B), actions] - rng.normal(0, 0.3, size=B)
    adv = rng.normal(size=B)
    ret = rng.normal(size=B)
    return logits, values, actions, old_logp, adv, ret


def test_gradient_check_full_loss():
    logits, values, actions, old_logp, adv, ret = tiny_logits_values()
    eps = 1e-6
    loss, dlogits, dvalues = ppo_loss_and_grads(logits, values, actions, old_logp, adv, ret,
                                                clip_eps=0.2, vf_coef=0.5, ent_coef=0.01)
    num = np.zeros_like(logits)
    for i in range(logits.shape[0]):
        for k in range(logits.shape[1]):
            lp = logits.copy(); lp[i, k] += eps
            lm = logits.copy(); lm[i, k] -= eps
            l1, _, _ = ppo_loss_and_grads(lp, values, actions, old_logp, adv, ret)
            l2, _, _ = ppo_loss_and_grads(lm, values, actions, old_logp, adv, ret)
            num[i, k] = (l1 - l2) / (2 * eps)
    np.testing.assert_allclose(dlogits, num, atol=1e-6, rtol=1e-4)

    numv = np.zeros_like(values)
    for i in range(values.shape[0]):
        vp = values.copy(); vp[i] += eps
        vm = values.copy(); vm[i] -= eps
        l1, _, _ = ppo_loss_and_grads(logits, vp, actions, old_logp, adv, ret)
        l2, _, _ = ppo_loss_and_grads(logits, vm, actions, old_logp, adv, ret)
        numv[i] = (l1 - l2) / (2 * eps)
    np.testing.assert_allclose(dvalues, numv, atol=1e-6, rtol=1e-4)


def test_constant_advantage_positive_direction():
    # with adv = +c and ratio near 1, policy gradient should push up the log-prob
    # of the taken action (logit increases along onehot(a) - p)
    B = 4
    logits = np.zeros((B, 2))
    values = np.zeros(B)
    actions = np.array([0, 1, 0, 1])
    old_logp = np.full(B, np.log(0.5))
    adv = np.full(B, 1.0)
    ret = np.zeros(B)
    _, dlogits, _ = ppo_loss_and_grads(logits, values, actions, old_logp, adv, ret, ent_coef=0.0)
    # moving along -grad must increase logp of the taken action
    for i in range(B):
        d = -dlogits[i]
        step = 1e-3
        l1, _, _ = ppo_loss_and_grads(logits.copy(), values, actions, old_logp, adv, ret, ent_coef=0.0)
        trial = logits.copy(); trial[i, actions[i]] += step
        l2, _, _ = ppo_loss_and_grads(trial, values, actions, old_logp, adv, ret, ent_coef=0.0)
        assert (l2 - l1) / step < 0  # loss decreases when logit of taken action rises


def test_clip_saturates_when_ratio_far_off():
    # adv>0 but the sampled logp is far above old: ratio > 1+eps caps the objective
    logits = np.array([[5.0, -5.0]])
    values = np.zeros(1)
    actions = np.array([0])
    old_logp = np.array([np.log(0.5)])  # true logp ~ 0 >> old + eps
    adv = np.array([1.0])
    ret = np.zeros(1)
    l_far, dlogits_far, _ = ppo_loss_and_grads(logits, values, actions, old_logp, adv, ret, ent_coef=0.0)
    # objective is clipped at (1+eps)*adv regardless of how good ratio gets
    assert np.isclose(l_far, -(1.2) * 1.0, atol=1e-6)
    # and the gradient through the clipped branch is zero
    assert np.allclose(dlogits_far, 0.0, atol=1e-12)


def test_entropy_term_prefers_uniform():
    logits = np.array([[3.0, -3.0]])
    values = np.zeros(1)
    actions = np.array([0])
    old_logp = np.log(np.array([np.exp(3) / (np.exp(3) + np.exp(-3))]))
    adv = np.zeros(1)
    ret = np.zeros(1)
    # with zero advantage, only value(=0) and entropy remain: loss = -ent_coef * H
    l_peaked, _, _ = ppo_loss_and_grads(logits, values, actions, old_logp, adv, ret, ent_coef=0.01)
    l_uniform, _, _ = ppo_loss_and_grads(np.zeros((1, 2)), values, actions,
                                         np.array([np.log(0.5)]), adv, ret, ent_coef=0.01)
    assert l_uniform < l_peaked  # uniform policy has higher entropy, lower loss


def test_actor_critic_backward_matches_finite_differences():
    net = ActorCritic(hidden=5, seed=3)
    rng = np.random.default_rng(9)
    x = rng.normal(size=(4, 4))
    actions = rng.integers(0, 2, size=4)
    adv = rng.normal(size=4)
    ret = rng.normal(size=4)
    # old_logp pinned to the current policy's logp so ratio == 1 sits strictly
    # inside the clip interval: finite differences across shared trunk weights
    # must not cross the clip kink at ratio = 1 +/- eps.
    logits0, _, _ = net.forward(x)
    from ppolab.ppo import log_softmax as _ls
    old_logp = _ls(logits0)[np.arange(4), actions]

    def loss_of(perturbed=None, key=None):
        # perturbed is a full copy of the parameter matrix with one element nudged
        if key is not None:
            orig = net.p[key]
            net.p[key] = perturbed
        logits, values, cache = net.forward(x)
        l, dlog, dval = ppo_loss_and_grads(logits, values, actions, old_logp, adv, ret)
        net.zero_grad()
        net.backward(cache, dlog, dval)
        if key is not None:
            net.p[key] = orig
        return l

    loss_of()  # populate grads
    eps = 1e-6
    for key in net.p:
        g = net.grads[key].copy()
        flat = net.p[key].ravel()
        gf = g.ravel()
        for j in range(0, flat.size, max(1, flat.size // 4)):
            plus = flat.copy(); plus[j] += eps
            minus = flat.copy(); minus[j] -= eps
            lp = loss_of(plus.reshape(net.p[key].shape), key)
            lm = loss_of(minus.reshape(net.p[key].shape), key)
            num = (lp - lm) / (2 * eps)
            assert np.isclose(gf[j], num, atol=1e-5, rtol=1e-3), (key, j, gf[j], num)
