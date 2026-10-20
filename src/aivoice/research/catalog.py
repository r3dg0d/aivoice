"""Research corpus: what each paper contributed to aivoice, and how (verified 2026-10-20).

Status vocabulary: directly-implemented | faithfully-reproduced | adapted | experimental |
investigated | blocked-weights | blocked-license | unverified.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Paper:
    key: str
    title: str
    venue: str
    url: str
    technique: str
    code: str
    weights: str
    license: str
    status: str
    in_aivoice: str
    bibtex: str


PAPERS: list[Paper] = [
    Paper(
        "crosslingual-f5tts2",
        "Cross-Lingual F5-TTS 2: A Simplified Framework for Language-Agnostic Voice Cloning",
        "arXiv 2609.15184 (2026)",
        "https://arxiv.org/abs/2609.15184",
        "transcript-free prompts via same-speaker synthetic pairs; silence-aware speaking-rate predictor",
        "github.com/QingyuLiu0521/Cross-Lingual_F5-TTS_2 (seen in search; not inspected: host blocked here)",
        "SRP checkpoint listed in the repo README (not verified)",
        "unverified",
        "adapted",
        "research/duration.py: speech-only (VAD) rate measurement at inference time; the learned predictor and "
        "the fine-tuning recipe are not reproduced",
        "@article{liu2026crosslingualf5tts2, title={Cross-Lingual F5-TTS 2: A Simplified Framework for Language-Agnostic Voice Cloning}, journal={arXiv preprint arXiv:2609.15184}, year={2026}}",
    ),
    Paper(
        "voxcpm2",
        "VoxCPM2: Tokenizer-Free TTS for Multilingual Speech Generation, Creative Voice Design, and True-to-Life Cloning",
        "arXiv 2606.06928 (2026); OpenBMB",
        "https://github.com/OpenBMB/VoxCPM",
        "2B tokenizer-free TTS, 30 languages, 48 kHz AudioVAE V2 output, reference-only cloning, voice design / style instructions",
        "pip voxcpm 2.0.3 (inspected)",
        "openbmb/VoxCPM2",
        "Apache-2.0 (code and weights, per official README)",
        "directly-implemented",
        "engines/research_engines.py VoxCPM2Engine via the official package in an isolated worker; not run here (no Hub access)",
        "@misc{voxcpm2_2026, title={VoxCPM2: Tokenizer-Free TTS for Multilingual Speech Generation, Creative Voice Design, and True-to-Life Cloning}, author={OpenBMB}, year={2026}, note={arXiv:2606.06928}}",
    ),
    Paper(
        "xvoice",
        "X-Voice: Enabling Everyone to Speak 30 Languages via Zero-Shot Cross-Lingual Voice Cloning",
        "arXiv 2605.05611 (2026)",
        "https://arxiv.org/abs/2605.05611",
        "0.4B flow matching on IPA, language IDs, decoupled scheduled CFG, transcript-free Stage 2 with SRP",
        "pip x-voice 0.1.1 (inspected); github.com/sunnyxrxrx/X-Voice",
        "XRXRX/X-Voice",
        "code MIT; weights: model card not verifiable here",
        "directly-implemented",
        "engines/research_engines.py XVoiceEngine via its Stage-2 CLI; research/phonetics.py emits its [main|lang] code-switch tags",
        "@article{xvoice2026, title={X-Voice: Enabling Everyone to Speak 30 Languages via Zero-Shot Cross-Lingual Voice Cloning}, journal={arXiv preprint arXiv:2605.05611}, year={2026}}",
    ),
    Paper(
        "confucius4",
        "Confucius4-TTS: Transcript-Free Cross-Lingual Zero-Shot TTS with a Learnable Speaker Encoder",
        "arXiv 2608.11650 (2026), NetEase Youdao",
        "https://arxiv.org/abs/2608.11650",
        "LLM text-to-semantic with a learnable speaker encoder on SSL features; flow-matching DiT semantic-to-acoustic",
        "github.com/netease-youdao/Confucius4-TTS (not on PyPI; not inspected: host blocked here)",
        "release stated in the abstract; not verified",
        "unverified",
        "investigated",
        "not integrated: no inspectable package or weights from this environment. The learnable speaker encoder is part "
        "of its own training and cannot be bolted onto other engines",
        "@article{wang2026confucius4, title={Confucius4-TTS: Transcript-Free Cross-Lingual Zero-Shot TTS with a Learnable Speaker Encoder}, journal={arXiv preprint arXiv:2608.11650}, year={2026}}",
    ),
    Paper(
        "fc-tts",
        "FC-TTS: Style and Timbre Control in Zero-Shot Text-to-Speech with Disentangled Speech Representations",
        "ACL 2026 (long), 2026.acl-long.173; arXiv 2605.24618",
        "https://aclanthology.org/2026.acl-long.173/",
        "separate timbre and style references over factorized (FACodec) representations",
        "no public code found",
        "none found",
        "n/a",
        "blocked-weights",
        "engines/base.py: SynthesisOptions.style_reference exists and is refused by every engine that cannot use it "
        "(none can today); no embedding blending is faked. Identity vs natural-language style is split natively for VoxCPM2",
        "@inproceedings{lee2026fctts, title={FC-TTS: Style and Timbre Control in Zero-Shot Text-to-Speech with Disentangled Speech Representations}, author={Lee, Yoonhyung and Park, Hyunsin and Park, Jinhwan and Lee, Jinkyu}, booktitle={Proceedings of ACL 2026}, pages={3772--3791}, year={2026}}",
    ),
    Paper(
        "voicestar",
        "VoiceStar: Robust Zero-Shot Autoregressive TTS with Duration Control and Extrapolation",
        "Findings of ACL 2026, 2026.findings-acl.570",
        "https://aclanthology.org/2026.findings-acl.570/",
        "PM-RoPE duration control, continuation-prompt mixed training, long-form extrapolation",
        "pip voicestar 0.1.0 (inspected); github.com/jasonppy/VoiceStar",
        "pyp1/VoiceStar",
        "code MIT, weights CC-BY-4.0",
        "directly-implemented",
        "VoiceStarEngine; its default duration estimate equals the reference length, so aivoice supplies a "
        "silence-aware plan (fusion with Cross-Lingual F5-TTS 2); research/longform.py",
        "@inproceedings{peng2026voicestar, title={VoiceStar: Robust Zero-Shot Autoregressive TTS with Duration Control and Extrapolation}, author={Peng, Puyuan and Zheng, Zhisheng and Li, Shang-Wen and Mohamed, Abdelrahman and Harwath, David}, booktitle={Findings of ACL 2026}, pages={11737--11754}, year={2026}}",
    ),
    Paper(
        "flexivoice",
        "FlexiVoice: Enabling Flexible Style Control in Zero-Shot TTS with Natural Language Instructions",
        "ICLR 2026",
        "https://flexi-voice.github.io/",
        "natural-language style instructions with zero-shot timbre, preference optimisation",
        "not verified",
        "not verified",
        "n/a",
        "adapted",
        "`--style` natural-language control exposed only for engines that implement it natively (VoxCPM2 instruction "
        "prefix, Drama 3 bracket direction); FlexiVoice itself is not integrated",
        "@inproceedings{chen2026flexivoice, title={FlexiVoice: Enabling Flexible Style Control in Zero-Shot TTS with Natural Language Instructions}, author={Chen, Dekun and Zhang, Xueyao and Wang, Yuancheng and Dai, Kenan and Ma, Li and Wu, Zhizheng}, booktitle={ICLR}, year={2026}}",
    ),
    Paper(
        "pfluxtts",
        "PFluxTTS: Hybrid Flow-Matching TTS with Robust Cross-Lingual Voice Cloning and Inference-Time Model Fusion",
        "arXiv 2602.04160, ICASSP 2026",
        "https://arxiv.org/abs/2602.04160",
        "inference-time vector-field fusion of duration-guided and alignment-free decoders; 48 kHz PeriodWave vocoder",
        "no code found",
        "none found",
        "n/a",
        "experimental",
        "research/flow_matching.py: fused-field ODE sampler with a strict state-space compatibility check; no installed "
        "engine pair is compatible, so it is never used in synthesis",
        "@inproceedings{pfluxtts2026, title={PFluxTTS: Hybrid Flow-Matching TTS with Robust Cross-Lingual Voice Cloning and Inference-Time Model Fusion}, booktitle={ICASSP}, year={2026}, note={arXiv:2602.04160}}",
    ),
    Paper(
        "adaptation",
        "Speaker-specific adaptation vs zero-shot TTS (EURASIP JASMP, doi 10.1186/s13636-026-00479-w)",
        "EURASIP J. Audio Speech Music Process. (2026), as cited in the brief",
        "https://link.springer.com/article/10.1186/s13636-026-00479-w",
        "trade-offs between zero-shot and parameter-efficient adaptation",
        "n/a",
        "n/a",
        "n/a",
        "unverified",
        "could not retrieve the article from this environment; aivoice's adaptation path (research/adaptation.py, "
        "VoxCPM LoRA manifests) follows VoxCPM's own documented fine-tuning workflow instead",
        "% unverified reference: doi 10.1186/s13636-026-00479-w",
    ),
    Paper(
        "meanvc2",
        "MeanVC2: Robust Low-Latency Streaming Zero-Shot Voice Conversion",
        "arXiv 2606.09050 (2026)",
        "https://arxiv.org/abs/2606.09050",
        "streaming zero-shot VC with mean flows, 40/120 ms chunks",
        "github.com/ASLP-lab/MeanVC2 (vendor checkout)",
        "ASLP-lab/MeanVC2",
        "README claims Apache-2.0; LICENSE file missing upstream (NOTICE)",
        "directly-implemented",
        "existing baseline; MeanVC2Engine adapter; the v2 reference preparation applies to its target clips too",
        "@article{ma2026meanvc2, title={MeanVC2: Robust Low-Latency Streaming Zero-Shot Voice Conversion}, author={Ma, Guobin and Xia, Yuxuan and Jiang, Yuepeng and Guo, Dake and Xie, Hanke and Hu, Jingbin and Wang, Yanbo and Xie, Lei and Zhu, Pengcheng}, journal={arXiv preprint arXiv:2606.09050}, year={2026}}",
    ),
    Paper(
        "ge2e",
        "Generalized End-to-End Loss for Speaker Verification",
        "ICASSP 2018",
        "https://arxiv.org/abs/1710.10467",
        "speaker-verification embeddings (used for evaluation only)",
        "Resemblyzer (Apache-2.0) pretrained.pt",
        "bundled in the resemblyzer wheel",
        "Apache-2.0",
        "faithfully-reproduced",
        "research/speaker_identity.py reproduces Resemblyzer's inference exactly (cosine 1.0000 vs its own encoder)",
        "@inproceedings{wan2018ge2e, title={Generalized End-to-End Loss for Speaker Verification}, author={Wan, Li and Wang, Quan and Papir, Alan and Moreno, Ignacio Lopez}, booktitle={ICASSP}, year={2018}}",
    ),
]


def by_key(key: str) -> Paper:
    for p in PAPERS:
        if p.key == key:
            return p
    raise KeyError(key)


def bibtex() -> str:
    return "\n\n".join(p.bibtex for p in PAPERS) + "\n"


def as_dicts() -> list[dict]:
    return [asdict(p) for p in PAPERS]
