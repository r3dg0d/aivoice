# Galaxy theme (shared by the aivoice and deepfake QuickShell apps)

Both widgets are drawn with the same small design system, taken from the installed rice
(`~/.config/quickshell/galaxy`, `galaxy-hub`). Each repository ships its own copy of the files below,
so either app runs without the other. There is no runtime cross-repository dependency. **Keep the
copies identical**; change both together.

| file | what it is |
|---|---|
| `Galaxy.qml` | `pragma Singleton` token set: palette, fonts, radii, spacing, motion |
| `GalaxyButton.qml` | the rice button (`primary` pill, `selected`, `danger`, `hint` tooltip) |
| `GalaxyPanel.qml` | glass card: rice panel gradient + thin luminous border, `hot` highlight |
| `GalaxySection.qml` | spaced small-caps heading, `✦  T I T L E` |
| `GalaxyStat.qml` | metric tile (label + Zerodot value); show `—` for values the backend did not report |
| `GalaxyPill.qml` | status pill: `ok` / `warn` / `err` / `busy` / `idle`, pulsing dot for ok/busy |
| `GalaxyField.qml`, `GalaxyCombo.qml`, `GalaxyToggle.qml`, `GalaxySlider.qml` | form controls |
| `GalaxyProgress.qml` | download/job progress with an indeterminate comet |
| `GalaxySpark.qml` | sparkline with violet fill (repaints only when data changes) |
| `GalaxyNav.qml` | sidebar entry with a glowing rail |
| `Starfield.qml` | static nebula + stars (+ optional window base gradient), painted once per resize |

## Tokens

| token | value | use |
|---|---|---|
| `void_` `base` `surface` `raised` `tile` | `#07060d` `#0b0914` `#0f0d1a` `#1a1630` `#120f22` | backgrounds, darkest to raised |
| `line` `edge` `edgeHot` | `#26223a` `#3a2f5c` `#5b3d99` | borders; `edgeHot` = active/live |
| `violet` `purple` `lavender` `magenta` | `#7c4dff` `#b46bff` `#d4b8ff` `#ff6bd5` | accents |
| `ink` `muted` `label` `dim` | `#f2f0ff` `#9a96b3` `#8a84a8` `#5d5878` | text |
| `ok` / `warn` / `err` (+ `*Bg`) | `#7dffa8` / `#ffd479` / `#ff7a9a` | status |
| `font` `mono` `icons` | Cellforge, Zerodot, JetBrainsMono Nerd Font | UI text, numbers, glyphs |
| `radiusWindow` `radiusPanel` `radiusTile` `radiusControl` | 22, 18, 14, 10 | corners |

## Rules

- **No gradients on rounded `Rectangle`s.** The scene graph interpolates them per triangle and a seam
  shows along the diagonal on the GPU renderer. Panels and the window background paint their
  gradient with a `Canvas` clipped to the rounded rect (`GalaxyPanel`, `Starfield { paintBase: true }`).
- **No per-frame work.** Canvas items repaint on resize or data change only; animations are limited
  to status dots and progress. The deepfake preview refreshes at 10 Hz, only while visible and live,
  double-buffered so PNG decoding never blocks the UI thread.
- **Never fabricate.** A metric the backend did not report renders as `—`. Controls exist only
  for actions the backend implements.
- Screenshots for review: both shells take `*_UI_SCREENSHOT=/path.png` (+ `*_UI_SCREENSHOT_QUIT=1`)
  and render offscreen with `QT_QPA_PLATFORM=offscreen`; the test suites do this for every page
  and fail on any QML warning.
