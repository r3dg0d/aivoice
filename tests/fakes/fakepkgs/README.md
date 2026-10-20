Stand-ins for `voxcpm` 2.0.3, `voicestar` 0.1.0 and `x-voice` 0.1.1 with the same call
signatures as the released packages (read from their wheels). They generate tones and log
every call to $AIVOICE_FAKE_LOG so tests can check what aivoice passed. No model inside.
