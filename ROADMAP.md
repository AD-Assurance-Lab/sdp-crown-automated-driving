# SDP-CROWN Automated Driving Verification Roadmap

This roadmap outlines future features, threat models, and architectural improvements for the neural network verification pipeline. It serves as a checklist for developers and autonomous agents to track long-term goals.

---

## 📋 Roadmap & Feature Checklist

### Phase 1: High-Priority Physics Models
- [ ] **Sun Glare & Nighttime High-Beams**
  - **Theory:** Parameterized Semantic Perturbation (SP) layer: $T(x, \epsilon) = \text{clamp}(x + \epsilon \cdot F, 0, 1)$ where $F$ is a pre-calibrated lens flare scattering pattern.
  - **Data:** Calibrate using Flare7K and Flare7K++ datasets.
  - **Status:** *Not Started* (See [Appendix Glare](file:///home/za/ad_assurance/sdp_crown_ieee_publication/main.tex#L370-L382) in the publication sources for math details).

- [ ] **Localized Lens Blinding (Adversarial Patch Verification)**
  - **Theory:** Spatial mask occlusion: $T(x, p) = (1 - M_{patch}) \odot x + M_{patch} \odot p$ where $p \in [p_{\min}, p_{\max}]$ tracks mud/grime color channels.
  - **Data:** Calibrate patch dimensions and sludgy color intervals using Valeo's WoodScape dataset.
  - **Status:** *Not Started* (See [Appendix Lens Blinding](file:///home/za/ad_assurance/sdp_crown_ieee_publication/main.tex#L385-L400) in the publication sources for math details).

### Phase 2: Dynamic & Environmental Extensions
- [ ] **Windshield Wipers & Rain Streaks**
  - Model localized spatial line occlusions that shift dynamically across sequential frames.
- [ ] **Advective Fog & Dust (Depth-Based Attenuation)**
  - Integrate Koschmieder's Law with scene depth-maps (using monocular depth estimators) to decay pixel visibility exponentially by distance rather than applying a global contrast drop.
- [ ] **Vehicle Dynamics Perturbations**
  - Verify stability under camera pitch and tilt variations caused by road surface bumps and vehicle payload changes.

### Phase 3: Engine & Tooling Optimizations
- [ ] **VRAM Scaling Resolution**
  - Investigate sub-graph partitioning or block-wise bound propagation to reduce the quadratic memory cost of dense matrix-mode convolution bounds tracking.
- [ ] **Closed-Loop Reachability Integration**
  - Map steering deviation bounds into 1D/2D kinematic vehicle models (e.g. speed, distance to lead car, lateral lane deviation) to certify closed-loop trajectory safety envelopes rather than static per-frame corridors.

---

## 🛠️ How to Contribute (For AI Agents & Humans)
1. **Calibration:** Add calibration scripts under `tools/` using the corresponding dataset (e.g., WoodScape, Flare7K).
2. **Layer Design:** Implement the PyTorch transformation layer in `semantic_layers.py`.
3. **Verification:** Wrap the target model in `verify_steering.py` and run verification.
4. **Update:** Check off the completed feature in this roadmap and commit.
