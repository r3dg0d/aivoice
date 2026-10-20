# Multilingual and cross-lingual synthesis

| Engine | Languages (per its documentation) | Cross-lingual cloning | Notes |
|---|---|---|---|
| VoxCPM2 | 30 | yes | No language tag needed |
| X-Voice | 30 (IPA-based) | yes, transcript-free | Code-switching via `[main\|lang]` tags |
| VoiceStar | English | no | |
| Fish S2-Pro / Drama 3 | multilingual | yes | |
| MeanVC2 | any (VC) | yes | Converts the source speech, whatever language it is in |

## In aivoice

- **`--language`.** Sets the target language. Without it, `research/phonetics.py` detects the language
  from the script (Latin, CJK, Cyrillic, Arabic, and so on).
- **`--priority multilingual`.** Prefers engines that support cross-lingual cloning.
- **Normalization.** Numbers, abbreviations and your `--lexicon` (JSON mapping a word to how it should be
  spoken) are expanded before synthesis. Number expansion is implemented for English only; other
  languages pass through unchanged.
- **Code-switching.** When text mixes scripts and the engine is X-Voice, each run is tagged
  `[main|lang]` as in X-Voice's inference README.
- **IPA.** `to_ipa` uses espeak-ng when it is installed, for inspection and syllable counting.

## Limits

None of the cross-lingual engines could be run here (no Hugging Face access, no GPU), so the cross-lingual
similarity and WER experiments (D) were not run. Language coverage figures come from each project's own
documentation.
