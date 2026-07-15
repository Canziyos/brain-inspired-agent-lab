# Brain Lab

A small brain-inspired agent laboratory. It currently combines a symbolic GridWorld, a Gymnasium-compatible environment backend, rule-based motivation and planning, and a small neural reward predictor.

The project is a practical sandbox for building toward embodied cognitive-agent experiments.

---

**Where Baby Vice is now:**

1. Symbolic planner still controls behavior.
2. Reward network predicts immediate reward.
3. Recurrent rate-neuron model predicts state changes and events.
4. Hidden neural state persists across steps.
5. Training now uses contiguous sequences.
6. One-step imagination evaluates candidate actions.
7. Hybrid utility combines policy purpose with predicted reward.
8. Metrics and CSV logging are working.
9. Baby Vice learns during a run, but forgets after restart.

**What remains next:**

1. Fix rare-event learning, especially `ATE_FOOD`.
2. Normalize outcome targets and improve sequence sampling.
3. Add explicit working memory signals.
4. Test whether recurrent memory genuinely beats reset state.
5. Save/load learned models between runs.
6. Move from one-step to two-step imagination.
7. Later compare rate neurons against spiking neurons.

So we are not randomly adding neural decorations. The path is:

```text
reliable prediction
→ useful memory
→ imagination
→ persistent learning
→ spiking comparison
```
---
