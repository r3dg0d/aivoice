# Voice profiles (experimental mode)

Stored under `~/.local/share/aivoice/tts_voices/<id>/profile.json`. Separate from
`aivoice voices`, the MeanVC2 conversion library; a library voice can seed a
profile (`voice add NAME --from-library VOICE --text "..."`).

Every profile requires `--i-have-rights`: an attestation that you own the voice or
have the person's explicit permission to reproduce it. aivoice records the
statement and time, and will not create, import or register a profile without it.

## Kinds

| Kind | Works with | What is stored | Created with |
|---|---|---|---|
| `local_voice` | fish-speech server (local or your remote) | reference audio + exact transcript | `voice add NAME --audio ref.wav --text "..."` |
| `local_voice_id` | fish-speech server | server `reference_id` + URL (+ optional audio copy) | `voice add-id NAME --reference-id ID [--register --audio --text]` |
| `hosted_fish_id` | hosted Fish Audio API only | fish.audio voice model id | `voice add-hosted NAME --reference-id ID [--check]` |

* **local_voice**: the audio and transcript are sent with each request
  (`references`); the server caches the encoded reference with
  `use_memory_cache: on`. 5 to 30 s of clean speech with an exact transcript
  works best; `voice info` flags clips under 3 s or over 60 s.
* **local_voice_id**: `--register` uploads the clip with
  `POST /v1/references/add` (fields `id`, `audio`, `text`), so later requests send
  only the id. `voice remove --server-delete` also deletes it on the server.
* **hosted_fish_id**: `--check` looks the id up with the official
  `GET /model/{id}`.

## Hosted ids vs local ids

They are **not interchangeable**. A hosted `reference_id` names a voice model
stored on fish.audio; a local server has its own `references/` store. The
official API offers no way to export a hosted voice's conditioning data for a
local Drama 3 server, and aivoice does not use private endpoints to try. Using a
hosted id with a local server (or the reverse) is refused with an explanation. To
use your own hosted voice locally, create a `local_voice` from your original,
authorized recording.

## Other commands

```bash
aivoice voice list | info NAME | use NAME | preview NAME [--text ...] [-o out.wav]
aivoice voice export NAME out.zip          # profile + reference audio
aivoice voice import out.zip --i-have-rights
aivoice voice remove NAME [--server-delete]
```

Imports are extracted with the same zip-slip/size-limited extractor as voice
downloads, and need a fresh attestation.

A profile can carry a `character` prompt used by the LLM's `character` mode
(`voice add ... --character "a calm, dry-witted narrator"`).
