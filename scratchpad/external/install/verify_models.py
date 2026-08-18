"""Load each installed external agent the way the framework does and prove the weights are real.

Mimics agents.py: import `agent_code.<name>.callbacks`, then chdir into the agent folder
(agents.py:305) before calling setup().
"""
import importlib, logging, os, sys, hashlib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

AGENTS = ["ext_xiaoxiae_bindist_v2", "ext_xiaoxiae_binary_v6",
          "ext_aielka_ql_atom", "ext_lijesse_featureeverything"]


class FakeSelf:
    train = False
    def __init__(self, name):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.CRITICAL)


for name in AGENTS:
    print("=" * 70)
    print(name)
    cb = importlib.import_module(f"agent_code.{name}.callbacks")
    fs = FakeSelf(name)
    prev = os.getcwd()
    os.chdir(os.path.join(ROOT, "agent_code", name))
    try:
        cb.setup(fs)
    finally:
        os.chdir(prev)

    if hasattr(fs, "model"):                      # xiaoxiae DQN
        sd = fs.model.state_dict()
        tot = sum(v.numel() for v in sd.values())
        print(f"  torch DQN, {len(sd)} tensors, {tot} params")
        for k, v in sd.items():
            print(f"    {k:28s} {tuple(v.shape)}  mean={v.float().mean():+.5f} std={v.float().std():.5f}")
        # fresh-init check: an untrained net has ~symmetric, small-std layers matching the
        # torch default init; a trained one drifts. Compare against a fresh DQN.
        fresh = cb.DQN(cb.FEATURE_VECTOR_SIZE, len(cb.ACTIONS), cb.LAYER_SIZES)
        fsd = fresh.state_dict()
        d = {k: float((sd[k].cpu().float() - fsd[k].cpu().float()).abs().mean()) for k in sd}
        print(f"  mean |trained - fresh_init| per tensor: "
              + ", ".join(f"{k.split('.')[-2]}.{k.split('.')[-1]}={v:.4f}" for k, v in d.items()))
        print(f"  file sha1 target-model.pt: "
              f"{hashlib.sha1(open(os.path.join(ROOT,'agent_code',name,'target-model.pt'),'rb').read()).hexdigest()[:16]}")

    if hasattr(fs, "agent"):                      # AI-ELka tabular
        q = fs.agent.q_table
        print(f"  q_table type={type(q).__name__} entries={len(q)}")
        k0 = next(iter(q))
        print(f"  key len={len(k0)} example={k0}")
        print(f"  value len={len(q[k0])} example={q[k0]}")
        allv = np.array([v for vals in q.values() for v in vals], dtype=float)
        nz = int((allv != 0).sum())
        print(f"  {allv.size} Q-values, {nz} nonzero ({100*nz/allv.size:.1f}%), "
              f"min={allv.min():+.4f} max={allv.max():+.4f} mean={allv.mean():+.4f}")
        rows_all_zero = sum(1 for vals in q.values() if not any(vals))
        print(f"  rows that are entirely zero (never updated): {rows_all_zero}/{len(q)}")

    if hasattr(fs, "policy_net"):                 # Li-Jesse duelling DQN
        net = fs.policy_net
        sd = net.state_dict()
        tot = sum(v.numel() for v in sd.values())
        print(f"  {type(net).__name__} from {type(net).__module__}, {len(sd)} tensors, {tot} params")
        for k, v in sd.items():
            print(f"    {k:34s} {tuple(v.shape)}  mean={v.float().mean():+.5f} std={v.float().std():.5f}")
        fresh = cb._model.DQN(34, 6)
        fsd = fresh.state_dict()
        d = np.mean([float((sd[k].cpu().float() - fsd[k].cpu().float()).abs().mean()) for k in sd])
        print(f"  mean |trained - fresh_init| over all tensors: {d:.4f}")
