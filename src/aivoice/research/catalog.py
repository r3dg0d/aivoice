"""Research corpus: what each paper contributed to aivoice, and how (verified 2026-10-20).

Status vocabulary: directly-implemented | faithfully-reproduced | adapted | experimental |
investigated | blocked-weights | blocked-license | unverified.

Evidence classes for live VC (0.6, verified 2026-10-21): peer-reviewed research | official research
implementation | community implementation | unvalidated experiment. Repositories were inspected at
the commits named in each entry; arXiv/Hugging Face pages were not reachable from the build
environment, so paper claims are quoted from search-indexed abstracts and labelled as such.
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
    # --- 0.6: live voice conversion ------------------------------------------------------------
    Paper(
        "xvc",
        "X-VC: Zero-shot Streaming Voice Conversion in Codec Space",
        "arXiv 2604.12456 (2026), SJTU X-LANCE",
        "https://arxiv.org/abs/2604.12456",
        "one-step conversion in a pretrained codec (SAC) latent space: GLM-4-Voice semantic tokens + acoustic latents, "
        "ERes2Net utterance speaker embedding via adaptive normalisation, target mel as frame-level condition; chunkwise "
        "streaming with overlap smoothing",
        "github.com/Jerrister/X-VC @ 49df8c5 (inspected: bins/infer_utils.py, models/codec/sac, configs/xvc.yaml), MIT",
        "huggingface.co/chenxie95/X-VC (xvc.pt; not reachable here) + zai-org/glm-4-voice-tokenizer + ModelScope ERes2Net",
        "code MIT; weights per model cards (not verified here)",
        "directly-implemented",
        "vc/stream_worker.py XVCHandler: upstream load path, conditions and chunk forward; SlidingWindowStreamer makes "
        "upstream's run_streaming incremental (sample-identical, unit-tested). Not run here: weights unreachable, no GPU. "
        "Never auto-selected until `aivoice benchmark vc` measures it ahead locally",
        "@article{zheng2026xvc, title={X-VC: Zero-shot Streaming Voice Conversion in Codec Space}, author={Zheng, Qixi and Zhao, Yuxiang and Chen, Xie and others}, journal={arXiv preprint arXiv:2604.12456}, year={2026}}",
    ),
    Paper(
        "rtvc",
        "RT-VC: Real-Time Zero-Shot Voice Conversion with Speech Articulatory Coding",
        "ACL 2025 System Demonstrations (2025.acl-demo.37); arXiv 2506.10289",
        "https://aclanthology.org/2025.acl-demo.37",
        "articulatory (EMA) content features + causal pitch tracker + speaker encoder + DDSP vocoder; reported 61.4 ms CPU latency",
        "github.com/Berkeley-Speech-Group/RT-VC @ 9d9a55d (inspected): model definitions and an offline notebook helper only",
        "none published",
        "no LICENSE file in the repository (all rights reserved by default)",
        "blocked-weights",
        "catalogue-only engine `rtvc` (BLOCKED_CODE): no checkpoints, no streaming loop, no license. Nothing is faked",
        "@inproceedings{liu2025rtvc, title={RT-VC: Real-Time Zero-Shot Voice Conversion with Speech Articulatory Coding}, author={Liu, Yisi and Wang, Chenyang and Kim, Hanjo and Khan, Raniya and Anumanchipalli, Gopala}, booktitle={Proceedings of ACL 2025: System Demonstrations}, year={2025}}",
    ),
    Paper(
        "zerovc",
        "Zero-VC: Zero-Lookahead Streaming Voice Conversion via Speaker Anonymization",
        "arXiv 2606.20218 (2026)",
        "https://arxiv.org/abs/2606.20218",
        "strictly causal zero-lookahead VC: distilled streaming w2v-BERT 2.0 content encoder, timbre encoder, streaming "
        "decoder; speaker anonymisation as a training-time perturbation against timbre leakage; 20 ms frames",
        "none found (demo page amphionteam.github.io/Zero-VC-demo only)",
        "none found",
        "n/a",
        "blocked-weights",
        "catalogue-only engine `zerovc` (RESEARCH_ONLY). Needed later: code + causal checkpoints + a license",
        "@article{zerovc2026, title={Zero-VC: Zero-Lookahead Streaming Voice Conversion via Speaker Anonymization}, journal={arXiv preprint arXiv:2606.20218}, year={2026}}",
    ),
    Paper(
        "tvtsyn",
        "TVTSyn: Content-Synchronous Time-Varying Timbre for Streaming Voice Conversion and Anonymization",
        "ICLR 2026; arXiv 2602.09389",
        "https://arxiv.org/abs/2602.09389",
        "Global Timbre Memory + content-synchronous, gated time-varying timbre instead of one static speaker vector; "
        "factorised VQ bottleneck; reported < 80 ms GPU latency",
        "no GitHub repository found; a Hugging Face page (warisqr007/TVTSyn) is indexed but was unreachable here",
        "warisqr007/TVTSyn (not inspected)",
        "CC BY-NC 4.0 per the indexed model page (non-commercial)",
        "unverified",
        "catalogue-only engine `tvtsyn`. Idea noted, not borrowed: static-vector vs time-varying timbre is an architectural "
        "property of the trained model and cannot be bolted onto another engine",
        "@inproceedings{quamer2026tvtsyn, title={TVTSyn: Content-Synchronous Time-Varying Timbre for Streaming Voice Conversion and Anonymization}, author={Quamer, Waris and Tseng, Mu-Ruei and Nasrallah, Ghady and Gutierrez-Osuna, Ricardo}, booktitle={ICLR}, year={2026}}",
    ),
    Paper(
        "stylestream",
        "StyleStream: Real-Time Zero-Shot Voice Style Conversion",
        "arXiv 2602.20113 (2026); Interspeech 2026",
        "https://arxiv.org/abs/2602.20113",
        "Destylizer (content with a tight bottleneck and text supervision) + Stylizer (DiT flow matching) for timbre, "
        "accent and emotion; ~1 s end-to-end latency (600 ms chunks)",
        "github.com/Berkeley-Speech-Group/StyleStream @ 38dd18c (inspected: inference/streaming.py)",
        "Louis0324/StyleStream (style encoder weights withheld; curated style embeddings only)",
        "Research, Educational and Not-for-Profit license (UC Regents): no commercial use",
        "directly-implemented",
        "engine `stylestream` (EXPERIMENTAL): upstream processing loop in vc/stream_worker.py; never chosen by `auto`; "
        "targets must be curated style folders (.wav + .npy). Not run here",
        "@article{liu2026stylestream, title={StyleStream: Real-Time Zero-Shot Voice Style Conversion}, author={Liu, Yisi and Lee, Nicholas and Anumanchipalli, Gopala}, journal={arXiv preprint arXiv:2602.20113}, year={2026}}",
    ),
    Paper(
        "streamvoiceanon",
        "Stream-Voice-Anon: Enhancing Utility of Real-Time Speaker Anonymization via Neural Audio Codec and Language Models",
        "ICASSP 2026; arXiv 2601.13948",
        "https://arxiv.org/abs/2601.13948",
        "causal codec language model (Firefly codec) with pseudo-speaker sampling and speaker-embedding noise mixing; "
        "~180 ms reported latency",
        "github.com/Plachtaa/StreamVoiceAnon @ 2017051 (inspected: evaluations/infer_arvc.py streaming API), Apache-2.0",
        "Plachta/StreamVoiceAnon",
        "code Apache-2.0; weights per model card",
        "directly-implemented",
        "engine `streamvoiceanon` (EXPERIMENTAL, CUDA only): setup_stream_caches / prefill_prompt / process_one_chunk in the "
        "worker; `alpha` < 1 is the anonymisation knob. Not run here",
        "@misc{kuzmin2026streamvoiceanon, title={Stream-Voice-Anon: Enhancing Utility of Real-Time Speaker Anonymization via Neural Audio Codec and Language Models}, author={Kuzmin, Nikita and Liu, Songting and Lee, Kong Aik and Chng, Eng Siong}, year={2026}, eprint={2601.13948}, archivePrefix={arXiv}}",
    ),
    Paper(
        "streamvoiceanon-plus",
        "StreamVoiceAnon+: Emotion-Preserving Streaming Speaker Anonymization via Frame-Level Acoustic Distillation",
        "Interspeech 2026; arXiv 2603.06079",
        "https://arxiv.org/abs/2603.06079",
        "fine-tunes StreamVoiceAnon with frame-level acoustic (emotion) distillation",
        "described in the StreamVoiceAnon README; models marked 'coming soon' at commit 2017051",
        "not released at the inspected commit",
        "n/a",
        "blocked-weights",
        "not integrated: no released weights",
        "@misc{kuzmin2026streamvoiceanonplus, title={StreamVoiceAnon+: Emotion-Preserving Streaming Speaker Anonymization via Frame-Level Acoustic Distillation}, author={Kuzmin, Nikita and Lee, Kong Aik and Chng, Eng Siong}, year={2026}, eprint={2603.06079}, archivePrefix={arXiv}}",
    ),
    Paper(
        "rvc",
        "Retrieval-based Voice Conversion (RVC v1/v2)",
        "community implementation (not peer-reviewed)",
        "https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI",
        "HuBERT/ContentVec features + FAISS index retrieval + RMVPE F0 + VITS/NSF-HiFiGAN decoder trained per voice",
        "rvc-python (runtime of aivoice's rt_worker.py); RVC-Project WebUI and IAHispano/Applio realtime code read for the "
        "block/SOLA design",
        "per-voice .pth/.index from their uploaders",
        "RVC-Project MIT; voice models per uploader",
        "directly-implemented",
        "engine `rvc`: rvc/rt_worker.py RvcRealtime (block + extra context + SOLA crossfade) behind the shared pipeline; "
        "also its own sounddevice path (`virtualmic --engine rvc`). .pth files are pickles: trusted models only",
        "% RVC has no paper; cite the repository: https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI",
    ),
    Paper(
        "laroche-dolson",
        "Improved phase vocoder time-scale modification of audio / New phase-vocoder techniques for pitch-shifting",
        "IEEE Trans. Speech Audio Process. 7(3), 1999; IEEE WASPAA 1999",
        "https://doi.org/10.1109/89.759041",
        "identity phase locking around spectral peaks; pitch shifting by moving peak regions",
        "n/a (algorithm)", "n/a", "n/a",
        "adapted",
        "modulation/dsp.py: identity phase locking on the excitation spectrum (experiment R: lower aperiodicity than plain "
        "propagation at 32/48 ms windows; cross-frame peak matching measured worse and was dropped)",
        "@article{laroche1999phase, title={Improved phase vocoder time-scale modification of audio}, author={Laroche, Jean and Dolson, Mark}, journal={IEEE Transactions on Speech and Audio Processing}, volume={7}, number={3}, pages={323--332}, year={1999}}",
    ),
    Paper(
        "true-envelope",
        "Efficient spectral envelope estimation and its application to pitch shifting and envelope preservation",
        "DAFx 2005",
        "https://www.dafx.de/paper-archive/2005/P_030.pdf",
        "iterative cepstral 'true envelope' for formant-preserving pitch shifting",
        "n/a (algorithm)", "n/a", "n/a",
        "adapted",
        "modulation/dsp.py _lifter_envelope (3 iterations, lifter order tied to the tracked F0); formant warping of that "
        "envelope is aivoice's own use of it",
        "@inproceedings{robel2005true, title={Efficient spectral envelope estimation and its application to pitch shifting and envelope preservation}, author={R{\\\"o}bel, Axel and Rodet, Xavier}, booktitle={Proceedings of DAFx}, year={2005}}",
    ),
    Paper(
        "voice-gender-acoustics",
        "Acoustic cues to perceived voice gender and body size (F0, formants / vocal-tract length, voice quality)",
        "Peterson & Barney 1952 (JASA); Hillenbrand et al. 1995 (JASA); Fitch & Giedd 1999 (JASA); "
        "Hillenbrand & Clark 2009 (Atten. Percept. Psychophys.); Skuk & Schweinberger 2014 (JSLHR)",
        "https://doi.org/10.3758/APP.71.5.1150",
        "perceived sex/masculinity depends on F0 and on formant frequencies (vocal-tract length), with smaller "
        "contributions from aperiodicity/spectrum level; pitch alone does not make a voice sound male",
        "n/a", "n/a", "n/a",
        "unverified",
        "not retrievable from this environment; used only as the rationale for testing formants alongside pitch. The "
        "defaults come from aivoice's own measurements (experiments M, N, L), not from these papers' numbers",
        "% unverified here (domain literature, not inspected in this session): Peterson & Barney 1952; Hillenbrand et al. 1995; Fitch & Giedd 1999; Hillenbrand & Clark 2009 doi:10.3758/APP.71.5.1150; Skuk & Schweinberger 2014 doi:10.1044/1092-4388(2013/12-0314)",
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
