# Steering-Assisted Dissent & Judge Decontamination for LLM Deliberation Committees

**A framework proposal and a set of exploratory experiments** for keeping *minority-but-valuable* viewpoints alive in multi-agent LLM decision committees, and for stopping an AI reward-model from silently training away decisiveness.

> ⚠️ **Status: research notes — proposals (§1–3) plus a first round of small-N measurements (§4).**
> §4 reports what happened when we actually ran the proposed experiments (July 2026): some proposals survived, some failed in instructive ways, and Proposition H⊥'s precondition was measured. Everything is still *single-model, tiny-N, greedy-decoded pilot evidence* — treat §4 as **pilot results**, not established findings. Please falsify.

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

## 4. Results update (July 2026) — what happened when we measured

We ran the §3 experiments on `Qwen2.5-14B-Instruct` (4-bit, greedy, `repeng` steering on layers −5…−17). Semi-synthetic FX decision cards (hold / reduce / exit) with expert-signed labels: 20 dev + 10 holdout + 10 literature-anchored. Scripts in [`/experiments`](experiments); the fictional card decks are included.

### 4.1 The decisive answer became reachable — but not the way §3.1/§3.4 predicted
- A "decisive" steering axis fixes **timid** hedging with surprising specificity: +4 flips a case whose own analysis already implied exit, while +10 does **not** overshoot hold/reduce controls. But the double-trap case stayed rounded at *every* strength — output coherence dies (+18, mojibake) before the verdict flips. **Addition has a ceiling.**
- §3.4-style **directional ablation of the hedge axis** (full projection, hedge-side clamp, project-and-set — via `repeng`'s `operator=` hook, no custom hooks needed) also failed to flip the verdict. But it *changed the justification*: with the axis removed, the judge switched to **majority-vote reasoning** ("everyone else says reduce — overwhelming support"). **Rounding has a double bottom**: a `risk-management→reduce` *mapping prior*, and a *conformity prior* underneath it.
- What actually cracked it: **structural decomposition** — ask the model only **factual binaries with no safe middle**, and let *code* own the verdict mapping: (1) *is the entry thesis alive or dead?* dead → exit. (2) if alive: *is the risk shape manageable by trimming, or unmanageable* (weekend-gap × evenly-balanced binary event)? unmanageable → exit. (3) else: hold vs reduce. This went 4/4 on the trap set, including the double-trap that no amount of steering or ablation could flip.

### 4.2 Controlled comparison (N=20 dev, then N=30 with holdout)
Against the *same* base model single-shot: accuracy 85% vs 80% (1 card — not significant), but **exit recall 100% vs 50%** and **rounding rate 0% vs 29%**. All four baseline errors were textbook rounding (exit→reduce) — the exact failure the framework targets. Error costs are asymmetric: the baseline's errors are the expensive tail (riding losers); the committee's residual errors are bounded opportunity cost (over-exiting consumed winners).

A domain expert's label correction then exposed an **architecture blind spot**: exit was reachable only through "thesis dead", yet an alive thesis can still demand exit when the risk is *unmanageable* (weekend gap over a coin-flip election — "you should not have this position at all"). Adding the manageability binary (step 2 above) opened this third route with **zero false positives** across 13 alive-thesis cases.

### 4.3 A protocol catalog nobody warned us about
- **Rounding migrates upstream.** Push it out of the verdict and it reappears inside any sub-classification that re-admits a middle-ish option — chosen at the first token, then rationalized post-hoc (e.g. *"the premise = the fact that we hold the position, which is already complete"* — motivated sophistry to reach the safe label).
- **Answer position flips a binary.** Answer-first (prefilled label token) collapsed to the safe pole (exit recall 25%); reasoning-first/label-last collapsed to the *opposite* pole. Single semantic binaries fall over whichever way you push; only *factual* binaries + code mapping held under pressure.
- **Extraction is a battlefield of its own.** Rare-kanji labels (破綻) get token-corrupted under steering — use robust tokens (A/B) on the last line. Label-last needs generous `max_new_tokens`: in one run the reasoning was *entirely correct* and the label was simply truncated away. And a naive substring decision-parser mistakes negations ("撤退することなく…") for verdicts — we ship the negation-aware parser (`decide.py`).

### 4.4 Proposition H⊥ — precondition measured
We extracted `v_val` (CAA on 30 signed cases: "verdict: correct" vs "verdict: wrong" minimal pairs, fixed suffix to control token identity) and measured angles at the control layers: **cos∠(v_val, v_hedge) = +0.036**, and |cos| ≤ 0.12 against all persona axes. **The precondition holds — hedging and value are geometrically separable.**

Combined with 4.1 this yields the interesting conclusion: geometry says hedge-ablation is *safe* (it cannot remove much validity), yet experiment says it is *insufficient* (rounding relocates). Both are consistent iff **the contamination is not a single linear direction at the decision layers** — it is implemented structurally (mapping + conformity priors). In-judge orthogonalization is therefore not enough on its own; the decontamination that worked was moving the verdict mapping *out of the model into code*, with the external outcome anchor of §3.3 as the remaining fallback — exactly the fallback the proposition reserved for the entangled case, reached here by a different road.

### 4.5 Literature-anchored deck (escalation / disposition / sizing)
Ten cards whose ground truth comes from validated paradigms instead of author assertion (Staw 1976 escalation-of-commitment, dollar auction, entrapment; Shefrin-Statman/Odean disposition effect, Weber-Camerer; Kelly-criterion sizing):
- **Disposition resistance 3/3** — genuine winners are held (the residual over-exit errors are confined to "thesis-consumed × in-profit" cases). Kelly / volatility resizing 2/2.
- **Staw 1976 beats a steered 2026 judge.** Under "personal responsibility + no formal disconfirmation yet", the judge declared the thesis alive and *itself advocated averaging down* — the 1976 self-justification bias reproduced verbatim inside the judge (the bare model too). Candidate fix: a time-stop clause in the thesis-death definition (past its expected realization window while moving against you = disconfirmed).
- **The anti-conformity instruction is double-edged**: in one entrapment case the judge dismissed a *correct* panel consensus as "short-term thinking" — the first case we observed of the committee machinery underperforming the bare model.
- ⚠️ **Reverse-contamination caveat**: textbook scenarios are almost certainly in the training data, so the bare model's strength on this deck (80%, same headline as the committee) is indistinguishable from memorization. The committee's edge shows on the less-recognizable semi-synthetic deck (+50pt exit recall); the two decks are different test regimes and must be read together.

### 4.6 Where this leaves the proposals
| §3 proposal | verdict from §4 |
|---|---|
| 3.1 steer the skeptic | Works for *analysis depth* and for de-timidifying; cannot flip a structural mapping prior. |
| 3.2 two-phase committee | Anonymous ideation + advocates + judge held up; "dissent must be answered" needs re-carving (anti-conformity over-fires). |
| 3.3 external outcome anchor | Still the fallback that geometry cannot replace; not yet wired in. |
| 3.4 H⊥ / ablation | Precondition **holds**; ablation **insufficient** — contamination is structural, not a direction. Decontaminate by moving the mapping into code. |
| 3.5 steering-distillation | Untested; now lower priority than the structural route. |

## 5. Reproduce

- `experiments/committee_lab*.py` — status-framing / conformity probes (flat vs status second round; embedding-diversity + skeptic-divergence metrics).
- `experiments/committee_exp.py` — the ideation-status experiment with a double-trap case and ground-truth scoring.
- `experiments/extract_*.py`, `sweep_strength.py` — RepE control-vector extraction (via [`repeng`](https://github.com/vgel/repeng)) and strength/layer sweeps; `extract_axes.py` (orthogonal persona basis), `extract_vval.py` (v_val + the H⊥ angle table + validity probe).
- **§4 pipeline**: `decide.py` (negation-aware verdict parser) · `committee_v5_structural.py` (structural decomposition, 4/4) · `committee_v4_ablation.py` (the ablation negative result / "double bottom") · `committee_c20.py` (N=20 controlled run) · `committee_cprime*.py` (the sub-classification protocol catalog, incl. both collapse directions) · `committee_q2prime.py` (third route: risk-manageability binary) · `committee_gtest.py` (literature deck end-to-end).
- **Card decks**: `committee_cases.py` (20 semi-synthetic dev), `cases_holdout.py` (10 holdout, expert-signed), `cases_gtheory.py` (10 literature-anchored). All fictional scenarios; calibration figures redacted.

Stack: `transformers` + `repeng` for steering; `Ollama` + `bge-m3` for the committee/embedding probes. Run counts are small; **raise N substantially before drawing conclusions.**

## 6. Limitations (please read before citing)

- Single base model (Qwen2.5-14B-Instruct, 4-bit), greedy decoding, fixed prompts; no seed/config sweeps; no statistical testing. §4 is pilot evidence, not established findings.
- The dev deck was iterated on repeatedly; the holdout deck was touched by three protocol iterations (one mechanical, but still). Confirmatory claims need untouched cards.
- Card labels are expert-signed or literature-anchored, but the *scenario translations* are author-written; the literature deck is likely **in the training data** (see the reverse-contamination caveat in §4.5).
- Metrics (embedding spread, decision-vote diversity) can disagree and measure different things.
- Proposition H⊥: the **precondition** is now measured (§4.4); the full decontamination claim (ablated-judge reward de-correlates from hedginess while tracking correctness) remains untested — and §4.1 suggests the more interesting failure mode is structural, not directional.
- Steering, judges, and reward models can all be gamed; nothing here is a safety guarantee.

## 7. Prior art this builds on
Representation Engineering (Zou et al., 2023); Contrastive Activation Addition (Rimsky et al., 2024); Persona Vectors (Chen et al., 2025); linear probes for strategic deception (Goldowsky-Dill et al., 2025); the `repeng` library (Vogel). Committee/conformity framing draws on Asch conformity and the "HiPPO" (highest-paid-person's-opinion) effect. The literature deck (§4.5) anchors on Staw (1976) escalation of commitment, Shubik's dollar auction, Rubin & Brockner's entrapment paradigm, the Shefrin–Statman (1985) / Odean (1998) disposition effect, Weber & Camerer (1998), and the Kelly criterion (1956).

## License
MIT. Offered freely as open research notes. If it helps you, test it harder than we did and report back.
