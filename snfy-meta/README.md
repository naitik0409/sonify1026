# snfy

**Alias package.** `pip install snfy` is exactly the same as
`pip install sonify1026` — it installs the real engine as a dependency.

```bash
pip install snfy
```

```python
import snfy

engine = snfy.SonifyEngine()
engine.synthesize("Hello there!", emotion=snfy.Emotion.WARM, output_path="out.wav")
```

The distribution is named **`snfy`**; the code, docs and changelog live in the
main project:

- PyPI: <https://pypi.org/project/sonify1026/>
- Source: <https://github.com/naitik0409/sonify1026>

Made by **Naitik**.
