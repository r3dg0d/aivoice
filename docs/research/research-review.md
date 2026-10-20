# Research review

Verified 2026-10-20 from this environment. Code and weights were inspected only where they could be fetched (PyPI wheels).
huggingface.co, GitHub release assets and most publisher sites were blocked, so no neural model was run here.
Machine-readable source: `src/aivoice/research/catalog.py` (`aivoice research list`, `aivoice research citations`).

## Status classes

| Status | Meaning |
|---|---|
| directly-implemented | The official package/API is integrated as an engine, against its real signatures |
| faithfully-reproduced | Re-implemented and checked numerically against the reference implementation |
| adapted | The paper's idea is applied in a different place or form than in the paper; the difference is stated |
| experimental | Implemented, not used by default, because a prerequisite does not exist yet |
| investigated | Read and assessed; not integrated, with the reason |
| blocked-weights | No public code or weights to integrate |
| blocked-license | Usable code/weights exist but the license forbids the use |
| unverified | The source could not be retrieved; claims about it are not relied on |

## Matrix

| Paper | Venue | Technique | Code | Weights | License | Status | In aivoice |
|---|---|---|---|---|---|---|---|
| [Cross-Lingual F5-TTS 2: A Simplified Framework for Language-Agnostic Voice Cloning](https://arxiv.org/abs/2609.15184) | arXiv 2609.15184 (2026) | transcript-free prompts via same-speaker synthetic pairs; silence-aware speaking-rate predictor | github.com/QingyuLiu0521/Cross-Lingual_F5-TTS_2 (seen in search; not inspected: host blocked here) | SRP checkpoint listed in the repo README (not verified) | unverified | **adapted** | research/duration.py: speech-only (VAD) rate measurement at inference time; the learned predictor and the fine-tuning recipe are not reproduced |
| [VoxCPM2: Tokenizer-Free TTS for Multilingual Speech Generation, Creative Voice Design, and True-to-Life Cloning](https://github.com/OpenBMB/VoxCPM) | arXiv 2606.06928 (2026); OpenBMB | 2B tokenizer-free TTS, 30 languages, 48 kHz AudioVAE V2 output, reference-only cloning, voice design / style instructions | pip voxcpm 2.0.3 (inspected) | openbmb/VoxCPM2 | Apache-2.0 (code and weights, per official README) | **directly-implemented** | engines/research_engines.py VoxCPM2Engine via the official package in an isolated worker; not run here (no Hub access) |
| [X-Voice: Enabling Everyone to Speak 30 Languages via Zero-Shot Cross-Lingual Voice Cloning](https://arxiv.org/abs/2605.05611) | arXiv 2605.05611 (2026) | 0.4B flow matching on IPA, language IDs, decoupled scheduled CFG, transcript-free Stage 2 with SRP | pip x-voice 0.1.1 (inspected); github.com/sunnyxrxrx/X-Voice | XRXRX/X-Voice | code MIT; weights: model card not verifiable here | **directly-implemented** | engines/research_engines.py XVoiceEngine via its Stage-2 CLI; research/phonetics.py emits its [main\|lang] code-switch tags |
| [Confucius4-TTS: Transcript-Free Cross-Lingual Zero-Shot TTS with a Learnable Speaker Encoder](https://arxiv.org/abs/2608.11650) | arXiv 2608.11650 (2026), NetEase Youdao | LLM text-to-semantic with a learnable speaker encoder on SSL features; flow-matching DiT semantic-to-acoustic | github.com/netease-youdao/Confucius4-TTS (not on PyPI; not inspected: host blocked here) | release stated in the abstract; not verified | unverified | **investigated** | not integrated: no inspectable package or weights from this environment. The learnable speaker encoder is part of its own training and cannot be bolted onto other engines |
| [FC-TTS: Style and Timbre Control in Zero-Shot Text-to-Speech with Disentangled Speech Representations](https://aclanthology.org/2026.acl-long.173/) | ACL 2026 (long), 2026.acl-long.173; arXiv 2605.24618 | separate timbre and style references over factorized (FACodec) representations | no public code found | none found | n/a | **blocked-weights** | engines/base.py: SynthesisOptions.style_reference exists and is refused by every engine that cannot use it (none can today); no embedding blending is faked. Identity vs natural-language style is split natively for VoxCPM2 |
| [VoiceStar: Robust Zero-Shot Autoregressive TTS with Duration Control and Extrapolation](https://aclanthology.org/2026.findings-acl.570/) | Findings of ACL 2026, 2026.findings-acl.570 | PM-RoPE duration control, continuation-prompt mixed training, long-form extrapolation | pip voicestar 0.1.0 (inspected); github.com/jasonppy/VoiceStar | pyp1/VoiceStar | code MIT, weights CC-BY-4.0 | **directly-implemented** | VoiceStarEngine; its default duration estimate equals the reference length, so aivoice supplies a silence-aware plan (fusion with Cross-Lingual F5-TTS 2); research/longform.py |
| [FlexiVoice: Enabling Flexible Style Control in Zero-Shot TTS with Natural Language Instructions](https://flexi-voice.github.io/) | ICLR 2026 | natural-language style instructions with zero-shot timbre, preference optimisation | not verified | not verified | n/a | **adapted** | `--style` natural-language control exposed only for engines that implement it natively (VoxCPM2 instruction prefix, Drama 3 bracket direction); FlexiVoice itself is not integrated |
| [PFluxTTS: Hybrid Flow-Matching TTS with Robust Cross-Lingual Voice Cloning and Inference-Time Model Fusion](https://arxiv.org/abs/2602.04160) | arXiv 2602.04160, ICASSP 2026 | inference-time vector-field fusion of duration-guided and alignment-free decoders; 48 kHz PeriodWave vocoder | no code found | none found | n/a | **experimental** | research/flow_matching.py: fused-field ODE sampler with a strict state-space compatibility check; no installed engine pair is compatible, so it is never used in synthesis |
| [Speaker-specific adaptation vs zero-shot TTS (EURASIP JASMP, doi 10.1186/s13636-026-00479-w)](https://link.springer.com/article/10.1186/s13636-026-00479-w) | EURASIP J. Audio Speech Music Process. (2026), as cited in the brief | trade-offs between zero-shot and parameter-efficient adaptation | n/a | n/a | n/a | **unverified** | could not retrieve the article from this environment; aivoice's adaptation path (research/adaptation.py, VoxCPM LoRA manifests) follows VoxCPM's own documented fine-tuning workflow instead |
| [MeanVC2: Robust Low-Latency Streaming Zero-Shot Voice Conversion](https://arxiv.org/abs/2606.09050) | arXiv 2606.09050 (2026) | streaming zero-shot VC with mean flows, 40/120 ms chunks | github.com/ASLP-lab/MeanVC2 (vendor checkout) | ASLP-lab/MeanVC2 | README claims Apache-2.0; LICENSE file missing upstream (NOTICE) | **directly-implemented** | existing baseline; MeanVC2Engine adapter; the v2 reference preparation applies to its target clips too |
| [Generalized End-to-End Loss for Speaker Verification](https://arxiv.org/abs/1710.10467) | ICASSP 2018 | speaker-verification embeddings (used for evaluation only) | Resemblyzer (Apache-2.0) pretrained.pt | bundled in the resemblyzer wheel | Apache-2.0 | **faithfully-reproduced** | research/speaker_identity.py reproduces Resemblyzer's inference exactly (cosine 1.0000 vs its own encoder) |

## Evidence for the adapted / experimental items

- **Silence-aware duration (Cross-Lingual F5-TTS 2 → VoiceStar, X-Voice):** experiment F, [experiments/F.md](experiments/F.md). Mean absolute duration error falls from 0.84 (rate over the whole clip) and 0.82 (VoiceStar 0.1.0 default) to 0.15 (transcript) / 0.14 (acoustic). Measured on 28 synthetic pairs.
- **Reference preparation:** experiment C, [experiments/C.md](experiments/C.md). Denoise beats plain preparation by +0.036 GE2E cosine (95% CI 0.020–0.053) and is kept as `auto`.
- **Long-form joins and enhancement:** experiment G, [experiments/G.md](experiments/G.md). Loudness spread across segments drops from 4.0 LU to 0.7 LU; injected clicks drop from 3 to 0.5 per clip.
- **Vector-field fusion (PFluxTTS):** `research/flow_matching.py` checks that two fields share a state space, sample rate, feature type and conditioning before fusing them. None of the installed engine pairs passes that check, so fusion is never used in synthesis. It is covered by unit tests on analytic fields only.
- **Style/timbre separation (FC-TTS, FlexiVoice):** a style instruction is passed only to engines that implement one natively (VoxCPM2). Any engine without native support refuses `--style-reference`; nothing is blended in embedding space.

## Not done, and why

- Experiments A, B, D, E and H were not run: they need model weights (Hugging Face blocked) or a GPU. Each is defined in `aivoice research experiments` and runs unchanged on a machine with the engines installed.
- Confucius4-TTS and the EURASIP adaptation article could not be retrieved, so nothing in aivoice depends on claims from them.
- No training was run. `aivoice clone adapt` validates datasets and writes VoxCPM LoRA manifests only.
