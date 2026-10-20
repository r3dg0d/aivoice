# Voice and style control

Identity (who is speaking) and style (how it is said) are separate inputs:

- **Identity** comes from the profile's reference audio.
- **Style** comes from `--style` (a natural-language instruction), `--emotion`, `--speed` and `--duration`.
  Each engine accepts only what it implements.

| Option | voxcpm2 | xvoice | voicestar | fish-server | fish-api |
|---|---|---|---|---|---|
| `--style "calm, slow"` | yes: `(calm, slow)text` prefix | refused | refused | Drama 3 only: `[calm, slow]` bracket direction | yes |
| `--style-reference F` | refused | refused | refused | refused | refused |
| `--duration S` | refused | yes | yes | refused | refused |
| `--speed X` | refused | yes | yes (scales the duration plan) | refused | refused |

"Refused" means aivoice raises `UnsupportedOption`. `--allow-unsupported` drops the option with a
warning instead; aivoice never silently ignores one. Run `aivoice engines info <name>` to see the
current capabilities.

## Why there is no style reference

FC-TTS and FlexiVoice show that timbre and style can be controlled separately, but only in models
trained with factorized representations. Mixing speaker embeddings from two clips in a model not trained
for it degrades identity without reliable style transfer. aivoice therefore refuses `--style-reference`
until an engine supports it natively.

## Real-time mode

The real-time path (Path B) keeps its existing expression compiler for Fish bracket tags. When an engine
is selected with `aivoice experimental tts use engine:voxcpm2 --voice <profile>`, the style setting
`tts.style` is passed as VoxCPM's instruction prefix.
