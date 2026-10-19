# Related work and novelty check

Searched 2026-10-19 (arXiv, OpenReview, Alignment Forum/LessWrong, lab blogs). Each listed item's page was opened unless it is marked *(unverified)*. "No close prior work found" means only that this search found none. It is not a claim of novelty.

## Background the ideas build on
- Schlag, Irie & Schmidhuber, *Linear Transformers Are Secretly Fast Weight Programmers*, ICML 2021, arXiv 2102.11174. This is Idea 3's Part A.
- von Oswald et al., *Transformers Learn In-Context by Gradient Descent*, ICML 2023, arXiv 2212.07677.
- Geshkovski, Letrouit, Polyanskiy & Rigollet, *A mathematical perspective on Transformers*, arXiv 2312.10794. Shows clustering, i.e. attractors, in weight-shared dynamics.
- Delétang et al., *Language Modeling Is Compression*, ICLR 2024, arXiv 2309.10668.

## Q1. Is affect over-represented in sub-threshold activity? — no close prior work found
- Engels, Riggs Smith & Tegmark, *Decomposing The Dark Matter of Sparse Autoencoders*, TMLR 2025, arXiv 2410.14670. Much of the SAE error is linearly predictable. It does not look at sentiment.
- wesg, *SAE reconstruction errors are (empirically) pathological*, Alignment Forum, 2024.
- Marks et al., *Sparse Feature Circuits*, ICLR 2025, arXiv 2403.19647. SAE error nodes matter causally.
- Tigges et al., *Linear Representations of Sentiment in LLMs*, arXiv 2310.15154.
- Sun, Stolfo et al., *Dense SAE Latents Are Features, Not Bugs*, NeurIPS 2025, arXiv 2506.15679.
- Reichman, Avsian & Heck, *Emotions Where Art Thou*, arXiv 2510.22042.
- *Decoding Dark Matter: Specialized SAEs for Interpreting Rare Concepts*, arXiv 2411.00743 *(unverified)*.

## Q2. Do many small writes add up coherently (vs a sign-shuffled null)? — no close prior work found for this statistic
- Heimersheim & Turner, *Residual stream norms grow exponentially over the forward pass*, Alignment Forum, 2023. Layer contributions are correlated rather than independent. This is the closest observation.
- Lad, Gurnee & Tegmark, *The Remarkable Robustness of LLMs: Stages of Inference?*, arXiv 2406.19384. Covers ensembling and suppression neurons.
- Janiak et al., *An Adversarial Example for Direct Logit Attribution*, arXiv 2310.07325 *(authors unverified)*. Shows targeted cancellation.

## Q3. One shared vector field or layer-specific dynamics? — partially done
- Lu et al., *Understanding and Improving Transformer From a Multi-Particle Dynamic System Point of View*, arXiv 1906.02762.
- Tong et al., *Neural ODE Transformers*, ICLR 2025, arXiv 2503.01329 *(authors unverified)*. Reports evidence against weight sharing, which makes it the closest prior result.
- Sarfati et al., *Lines of Thought in Large Language Models*, ICLR 2025, arXiv 2410.01545.
- Sun et al., *Transformer Layers as Painters*, arXiv 2407.09298.
- Bae et al., *Relaxed Recursive Transformers*, ICLR 2025, arXiv 2410.20672.
- Belrose et al., *Tuned Lens*, arXiv 2303.08112.
- Jastrzębski et al., *Residual Connections Encourage Iterative Inference*, ICLR 2018 (ResNets).
- **Still open:** fitting one layer-independent field to a frozen model's actual residual updates, and comparing it with layer-specific fits on held-out error.

## Q4. Does compression predict low-rank emotion / theory-of-mind representations? — partially done
- Sofroniew et al. (Anthropic), *Emotion Concepts and their Function in a Large Language Model*, arXiv 2604.07729. The top PCs are valence and arousal.
- Choi & Weber, *Latent Structure of Affective Representations in LLMs*, arXiv 2604.07382. Finds a participation ratio of about 9–17, "diffuse rather than conclusively low-dimensional". **This partly cuts against a low-rank claim.**
- Sun et al., *Valence-Arousal Subspace in LLMs*, arXiv 2604.03147.
- Zhang & Zhong, *Decoding Emotion in the Deep*, arXiv 2510.04064 (preprint).
- Zhu et al., *Language Models Represent Beliefs of Self and Others*, arXiv 2402.18496.
- **Still open:** an MDL account of *why* the rank should be low, and rank measured across model scale or training checkpoints.

## Q5. Base vs instruct and the quiet tail (Phase 4) — no close prior work found
Searched 2026-10-20. Items not opened are marked *(unverified)*.
- **The closest specific comparison:** none found. No study we found ablates small-magnitude MLP activations against an energy-matched mid-magnitude control, or compares that between base and instruct models. Our wording: "to our knowledge, not previously reported", for one model pair.
- **Small activations are prunable (base models only):** TEAL (Liu et al., ICLR 2025, arXiv 2408.14690) and CATS (Lee et al., arXiv 2404.08763). These are consistent with our small *absolute* costs (quiet deletion costs about 0.06 nats in Qwen3-Base). Our claim is about cost *per unit energy* relative to mid-tier activations, and the base vs instruct contrast. Neither work evaluates instruct models.
- **Magnitude statistics after instruction tuning:** Sun et al., *Massive Activations in Large Language Models*, COLM 2024 (arXiv 2402.17762). Covers the extreme top of the distribution, not the small tail.
- **Base/chat representation transfer:** Kissane et al., *SAEs (usually) Transfer Between Base and Chat Models*, Alignment Forum 2024. Transfer fails for Gemma v1, whose base and chat weights are unusually far apart. Possibly relevant to our Gemma nulls.
- **Model diffing:** Lindsey et al., *Sparse Crosscoders*, transformer-circuits 2024. Minder et al., *Overcoming Sparsity Artifacts in Crosscoders to Interpret Chat-Tuning*, NeurIPS 2025 (arXiv 2504.02922).
- **Post-training reweights existing mechanisms:** Prakash et al., *Fine-Tuning Enhances Existing Mechanisms*, ICLR 2024 (arXiv 2402.14811). Du et al., *How Post-Training Reshapes LLMs*, COLM 2025 (arXiv 2504.02904). Mukherjee et al., *RL Finetunes Small Subnetworks in LLMs*, NeurIPS 2025 (arXiv 2505.11711).
- **Dense or low-magnitude signals are functional:** Sun, Stolfo et al., *Dense SAE Latents Are Features, Not Bugs* (above).
- **Post-training and confidence/diversity (behavioural):**
  - Kirk et al., *Understanding the Effects of RLHF on LLM Generalisation and Diversity*, arXiv 2310.06452.
  - Huang et al., *The Alignment Paradox*, arXiv 2609.32617.
  - Stolfo et al., *Confidence Regulation Neurons in Language Models*, arXiv 2406.16254 *(unverified)*.
  - The GPT-4 Technical Report's calibration figure *(unverified in the body)*.
  - We did **not** measure calibration or entropy. Any link to these is a hypothesis.
