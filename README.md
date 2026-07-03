# Steering-Assisted Dissent & Judge Decontamination for LLM Deliberation Committees

**A framework proposal and a set of exploratory experiments** for keeping *minority-but-valuable* viewpoints alive in multi-agent LLM decision committees, and for stopping an AI reward-model from silently training away decisiveness.

> ⚠️ **Status: research notes and proposals — NOT validated results.**
> Every empirical observation below comes from *small, under-powered, single-model* runs with **no reproducibility controls and insufficient trial counts**. Treat them as **illustrations of a hypothesis**, not evidence. Numbers are anecdotes. The value here (if any) is the **framing, the formal proposition, and the reproducible harness** — offered so that others with more compute can test them properly. Please falsify.

**Keywords:** representation-engineering · activation-steering · control-vectors · LLM-as-a-judge · RLAIF · reward-hacking · Goodhart · multi-agent-committee · deliberation · conformity · Asch · HiPPO · dissent · deception-detection · concept-ablation · orthogonal-reward · preference-optimization · DPO · steering-distillation · mode-collapse

---

## 1. The problem: prompted "persona committees" collapse to a hedged median

A popular pattern is to run one base LLM under several system prompts ("bull", "bear", "skeptic", …), collect their opinions, and have a moderator synthesize a decision. The intuition is *diversity of viewpoints*.

Our exploratory observation (single 14B instruct model, FX-decision toy task) is that this **diversity is largely illusory**:

- All personas share the **same RLHF-induced prior toward a safe, balanced, well-hedged answer**. Under interaction they regress to a common median (e.g. "reduce the position") and refuse decisive minority calls (e.g. "exit fully").
- A prompt like *"you are a reformed con-artist geopolitical analyst; read who benefits, distrust the surface"* **does not** reliably produce a genuinely divergent decision. The persona talks differently but **votes with the median**.
- Embedding-space diversity of the opinions (bge-m3 cosine spread) **collapses** after a round of mutual visibility.

We call this **median-seeking / "rounding" (丸め)**. It is the central obstacle: *you cannot buy viewpoint diversity from one aligned base model with prompts alone.*

## 2. Exploratory experiments (illustrative, under-powered)

Harness: one base model (`qwen2.5:14b` via Ollama) as 5 personas + moderator; `bge-m3` embeddings for diversity metrics. Scripts in [`/experiments`](experiments).

| Probe | What we looked at | What we *saw* (n small, noisy) |
|---|---|---|
| Status framing | anonymous-equal ("flat") vs explicit-seniority ("badge") second round | Effect **small and metric-dependent**: decision-vote diversity and embedding-spread pointed in *opposite* directions. A dramatic single-run "inversion" **did not survive** an n=6 replication. |
| Skeptic silencing | does giving one agent seniority mute the dissenter? | Inconclusive — under-powered (only ~4/6 runs produced a dissenter at all). |
| Reachability of a decisive answer | a "double-trap" case whose sharp reading implies a decisive exit | The base model **could not reach** the decisive answer; it defaulted to the hedged middle → accuracy ≈ 0, no discrimination. |

**Honest takeaways:** (a) the harness works and is cheap to run; (b) **status effects, if real, are small — you need N ≥ ~20 per cell** before claiming anything; (c) more importantly, the *instrument* (a prompted committee on one aligned base) is **too dull** to test the interesting questions. This motivates the proposals below.

## 3. Proposals (the actual contribution)

### 3.1 Steer the skeptic instead of prompting it
Representation Engineering (RepE) / activation steering lets you extract a linear "read-against-the-grain / deception-detection" direction from contrastive prompts and **add it at inference**. Because it operates on activations, it can push **past the RLHF hedging prior that prompts cannot move**. Proposal: give the committee's skeptic a steered dissent brace so it can hold a decisive minority position under group pressure.

### 3.2 Two-phase committee (status where it helps, not where it hurts)
- **Ideation phase = status-free / anonymous**, skeptic speaks first (before anchoring) — to prevent anticipatory conformity (Asch / HiPPO).
- **Decision phase = weighted**, but weight by **calibrated track record on verifiable claims, not by badge**. Status is only a weak tie-breaking prior.
- **"Dissent must be answered"**: the moderator must steelman and explicitly rebut the minority view before overruling it — not silently bury it.
- Protect a **floor weight for a designated dissenter** so calibration on *small* claims cannot down-weight the specialist whose value is *tail-risk / narrative* detection.

### 3.3 Reward design for "reward answers that exceed expectation"
Rewarding *surprise alone* reward-hacks into confident nonsense (novelty ≠ value; "low quality is not weak, it is harmful"). Use a **product of three terms**:

```
reward  =  surprise (mechanical)  ×  outcome-correct (external ground truth)  ×  process-sound (judge)
```
- **surprise**: measured *mechanically* (low probability under the base model / distance from the base model's own answer) — never asked of an LLM, which conflates "surprising" with "wrong".
- **outcome-correct**: an **external, verifiable** signal (e.g. a back-test against realized outcomes) — this lives *outside* any judge's representation.
- **process-sound**: an LLM judge, restricted to grading the *reasoning chain*, with a rubric that explicitly forbids treating "hedging" as quality.

Keep a **KL anchor to the base** and refresh/ensemble the judge to resist Goodhart / mode-collapse under long training.

### 3.4 Proposition **H⊥** — decontaminating an LLM judge by orthogonality
An LLM judge shares the same hedging prior, so used naively it **rewards the hedged median and penalizes decisive surprise — training *more* hedging, not less.** Formalize the fix geometrically.

> **Proposition H⊥.** Model the judge's reward on a response representation φ(x) as
> **R(x) ≈ α·⟨φ(x), v_val⟩ + β·⟨φ(x), v_hedge⟩ + ε**,
> where `v_val` is the value/quality direction and `v_hedge` a "hedging" direction. The second term is *hedging contamination*. **Directionally ablating `v_hedge` from φ** (projecting onto its orthogonal complement) removes it, so the reward becomes orthogonal to hedging. Hence
> **contamination ∝ cos∠(reward-gradient, v_hedge)**, and driving this toward 0 is a *measurable* decontamination objective.
> **Precondition:** `v_hedge ⊥ v_val` must be *achievable* — i.e. hedging and value must be geometrically separable. **Measure the angle first.** If entangled (near-parallel), in-judge orthogonalization is lossy and one must fall back to the external outcome signal (which is orthogonal *by construction*, living outside the judge's representation).

Both `v_hedge` (hedged↔decisive contrast) and `v_val` (ground-truth-correct↔incorrect reasoning contrast) are extractable with the same RepE machinery. The first concrete experiment is simply: **estimate `cos∠(v_hedge, v_val)`; if large, ablate `v_hedge` from the judge and check that its scores de-correlate from response "hedginess" while still tracking correctness.**

### 3.5 Steering-distillation (make the change permanent)
Activation steering is a cheap, reversible *prototype* of what training would bake in. Pipeline:
1. Generate "beyond-expectation" candidates with the dissent steering vector.
2. **Filter by external ground truth** (surprising *and* later-correct).
3. Train (SFT/DPO, LoRA-scale) with survivors as `chosen`, hedged/median outputs as `rejected`.
4. → the steered disposition is **distilled into weights**; no prompt needed at inference.

Caveat: training shifts **disposition** (willingness to be decisive/surprising) more than **capability** (actually finding the right surprising answer, which is bounded by model size).

## 4. Reproduce

- `experiments/committee_lab*.py` — status-framing / conformity probes (flat vs status second round; embedding-diversity + skeptic-divergence metrics).
- `experiments/committee_exp.py` — the ideation-status experiment with a double-trap case and ground-truth scoring.
- `experiments/extract_*.py`, `sweep_strength.py` — RepE control-vector extraction (via [`repeng`](https://github.com/vgel/repeng)) and strength/layer sweeps.

Stack: `transformers` + `repeng` for steering; `Ollama` + `bge-m3` for the committee/embedding probes. The decision task is a **fictional** FX scenario ("Country X") used purely as an illustrative testbed. Run counts in the scripts are deliberately small; **raise N substantially before drawing conclusions.**

## 5. Limitations (please read before citing)

- Single base model; tiny N; fixed prompts; no seed/config sweeps; no statistical testing. **Nothing here is established.**
- Metrics (embedding spread, decision-vote diversity) can disagree and measure different things.
- "Ground truth" for the toy task is author-asserted.
- Proposition H⊥ is **unverified**; its precondition (separability of hedge vs value) is an open empirical question.
- Steering, judges, and reward models can all be gamed; nothing here is a safety guarantee.

## 6. Prior art this builds on
Representation Engineering (Zou et al., 2023); Contrastive Activation Addition (Rimsky et al., 2024); Persona Vectors (Chen et al., 2025); linear probes for strategic deception (Goldowsky-Dill et al., 2025); the `repeng` library (Vogel). Committee/conformity framing draws on Asch conformity and the "HiPPO" (highest-paid-person's-opinion) effect.

## License
MIT. Offered freely as open research notes. If it helps you, test it harder than we did and report back.
