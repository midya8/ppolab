"""PPO training loop for CartPole, pure NumPy, end to end."""
import argparse
import json
import os
import time

import numpy as np

from .cartpole import CartPole
from .nets import ActorCritic
from .optim import Adam
from .ppo import compute_gae, log_softmax, ppo_loss_and_grads, softmax


def evaluate(net, seed=12345, episodes=20, max_steps=500):
    env = CartPole(seed=seed, max_steps=max_steps)
    returns = []
    for _ in range(episodes):
        obs = env.reset()
        total, done = 0.0, False
        while not done:
            logits, _, _ = net.forward(obs[None, :])
            a = int(np.argmax(logits[0]))
            obs, r, terminated, truncated = env.step(a)
            total += r
            done = terminated or truncated
        returns.append(total)
    return float(np.mean(returns)), float(np.std(returns)), returns


def rollout(net, env, horizon, rng, gamma=0.99):
    """Compatible seven-array adapter with corrected reset boundaries.

    Returned rewards add gamma*V(actual final state) at truncation only;
    returned dones cut BOTH resets. Feed these into legacy compute_gae.
    Raw environment rewards/flags live in boundaries.collect_rollout.
    gamma must match compute_gae's gamma. Training uses explicit transitions.
    """
    from .boundaries import collect_rollout
    b = collect_rollout(net, env, horizon, rng)
    rewards = b["rewards"] + gamma * b["next_values"] * b["truncated"]
    done = b["terminated"] | b["truncated"]
    return (b["observations"], b["actions"], b["old_logp"], rewards, done,
            b["values"], float(b["next_values"][-1]))


def update(net, opt, obs, act, logp_old, adv, ret, clip_eps=0.2, vf_coef=0.5,
           ent_coef=0.01, epochs=10, minibatch=256, rng=None):
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    n = len(obs)
    idx = np.arange(n)
    stats = []
    for _ in range(epochs):
        rng.shuffle(idx)
        for start in range(0, n, minibatch):
            mb = idx[start:start + minibatch]
            logits, values, cache = net.forward(obs[mb])
            loss, dlogits, dvalues = ppo_loss_and_grads(
                logits, values, act[mb], logp_old[mb], adv[mb], ret[mb],
                clip_eps=clip_eps, vf_coef=vf_coef, ent_coef=ent_coef)
            net.zero_grad()
            net.backward(cache, dlogits, dvalues)
            opt.step(net.grads)
            stats.append(loss)
    return float(np.mean(stats))


def render_gif(net, path, seed=777, fps=50, frame_every=4, max_frames=130):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image

    env = CartPole(seed=seed)
    env.reset()
    frames = []
    fig, ax = plt.subplots(figsize=(6, 4), dpi=80)
    plt.close(fig)
    for step in range(min(env.max_steps, frame_every * max_frames)):
        logits, _, _ = net.forward(env.state[None, :])
        a = int(np.argmax(logits[0]))
        env.step(a)
        if step % frame_every == 0:
            x, _, theta, _ = env.state
            fig, ax = plt.subplots(figsize=(6, 4), dpi=80)
            ax.set_xlim(-2.6, 2.6)
            ax.set_ylim(-0.6, 1.4)
            ax.axhline(0, color="#888", lw=2)
            ax.add_patch(Rectangle((x - 0.25, -0.1), 0.5, 0.2, color="#2563eb"))
            pole_len, px, py = 1.0, x, 0.1
            ax.plot([px, px + pole_len * np.sin(theta)], [py, py + pole_len * np.cos(theta)],
                    color="#111", lw=5, solid_capstyle="round")
            ax.set_aspect("equal")
            ax.axis("off")
            fig.canvas.draw()
            img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
            frames.append(Image.fromarray(img))
            plt.close(fig)
            if env.last_termination == "terminated":
                break
    if frames:
        frames[0].save(path, save_all=True, append_images=frames[1:],
                       duration=int(1000 / fps), loop=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="assets")
    ap.add_argument("--horizon", type=int, default=2048)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    env = CartPole(seed=args.seed)
    net = ActorCritic(seed=args.seed)
    opt = Adam(net.p, lr=3e-4)
    rng = np.random.default_rng(args.seed + 1)

    curve = []
    t0 = time.time()
    for it in range(1, args.iters + 1):
        from .boundaries import collect_rollout, compute_gae_transitions
        batch = collect_rollout(net, env, args.horizon, rng)
        obs, act, logp_old, rew = [batch[k] for k in ("observations", "actions", "old_logp", "rewards")]
        adv, ret = compute_gae_transitions(rew, batch["values"], batch["next_values"],
            batch["terminated"], batch["truncated"], gamma=0.99, lam=0.95)
        loss = update(net, opt, obs, act, logp_old, adv, ret, rng=rng)
        if it % 2 == 0 or it == args.iters:
            mean, std, _ = evaluate(net, episodes=10)
            curve.append({"iter": it, "eval_mean": mean, "eval_std": std,
                          "rollout_reward": float(rew.sum()), "loss": loss})
            print(f"iter {it:3d}  rollout_return {rew.sum():7.1f}  eval {mean:6.1f} +/- {std:5.1f}  loss {loss:.4f}", flush=True)
            if mean >= 490.0:
                print("solved; stopping early", flush=True)
                break
    train_seconds = time.time() - t0
    net.save(os.path.join(args.out, "policy.npz"))

    mean, std, _ = evaluate(net, episodes=20)
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump({"final_eval_mean_20ep": mean, "final_eval_std_20ep": std,
                   "train_seconds": train_seconds, "curve": curve}, f, indent=2)
    print(f"final eval: {mean:.1f} +/- {std:.1f} over 20 episodes ({train_seconds:.0f}s train)", flush=True)

    # learning-curve plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    its = [c["iter"] for c in curve]
    vals = [c["eval_mean"] for c in curve]
    fig, ax = plt.subplots(figsize=(7, 4), dpi=110)
    ax.plot(its, vals, marker="o", ms=3, color="#2563eb")
    ax.axhline(500, color="#16a34a", ls="--", lw=1, label="max (500)")
    ax.set_xlabel("PPO iteration")
    ax.set_ylabel("mean eval return (10 episodes, greedy)")
    ax.set_title("PPO from scratch on CartPole: learning curve")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "learning_curve.png"))
    render_gif(net, os.path.join(args.out, "cartpole_demo.gif"))


if __name__ == "__main__":
    main()
