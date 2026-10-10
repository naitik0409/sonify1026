# Publishing `sonify1026` to PyPI — full guide

Distribution name: **`sonify1026`** (what people `pip install`)
Alias distribution: **`snfy`** (thin meta-package that installs `sonify1026`)
Import name: **`snfy`** (what people write in code)

---

## 0. Before you start: two things you cannot undo

1. **The name `sonify1026` is permanent.** Once you upload, nobody can ever claim
   it and you cannot rename or delete the project. Claim it early.
2. **Every version is permanent.** If you upload `0.1.0` with a bug, you must
   release `0.1.1` — you cannot re-upload `0.1.0`.

You need: a PyPI account, a verified email, and ~15 minutes.

---

## 1. Your details (already set)

`pyproject.toml` already carries the real values:

```toml
authors = [
    {name = "Naitik", email = "sanjeevshrivastava898@gmail.com"},
]
maintainers = [
    {name = "Naitik", email = "sanjeevshrivastava898@gmail.com"},
]

[project.urls]
Homepage = "https://github.com/naitik0409/sonify1026"
Source   = "https://github.com/naitik0409/sonify1026"
Issues   = "https://github.com/naitik0409/sonify1026/issues"
```

---

## 2. Create the PyPI account and token

1. Go to <https://pypi.org/account/register/> and register.
2. **Verify your email address** (PyPI will email you a link). Until you do,
   you cannot upload.
3. Sign in, then open <https://pypi.org/manage/account/token/>.
4. Click **"Add token"**.
   - **Token name:** e.g. `sonify1026-upload`
   - **Scope:** choose **"Entire project"** if the project does not exist yet —
     PyPI will attach this token to `sonify1026` the first time you upload.
     (You can also choose a specific project afterwards and create a new token.)
   - Set an expiry if you want a security reminder.
5. **Copy the token now** — it starts with `pypi-` and is shown only once.

> Never share or commit this token. It is a password.

---

## 3. Tell twine your credentials

Create the file `%USERPROFILE%\.pypirc` (i.e. `C:\Users\<you>\.pypirc`):

```ini
[distutils]
index-sources =
    pypi

[pypi]
username = __token__
password = pypi-AgEIcHlwaS5vcmc...YOUR-TOKEN-HERE...
```

The username is literally the text `__token__` — not your username.

<details>
<summary>Prefer not to create a file? Use environment variables instead</summary>

```powershell
$env:TWINE_USERNAME = "__token__"
$env:TWINE_PASSWORD = "pypi-AgEIcHlwaS5v..."
```
This only lasts for the current terminal window, which is safer on shared machines.
</details>

---

## 4. Install the tooling

```bash
python -m pip install --upgrade build twine
```

---

## 5. Dry run on TestPyPI first (recommended)

TestPyPI is a separate sandbox — mistakes there are harmless.

1. Register at <https://test.pypi.org/account/register/> (separate account) and
   verify the email.
2. Create an API token there too, and add a `[testpypi]` section to `.pypirc`:

```ini
[testpypi]
username = __token__
password = pypi-AgEIcHlwaS5vcmc...TEST-TOKEN-HERE...
```

3. Build and upload to TestPyPI:

```bash
Remove-Item dist -Recurse -Force     # Windows PowerShell
python -m build
python -m twine check dist/*
python -m twine upload --repository testpypi dist/*
```

4. Install it from TestPyPI in a throwaway environment:

```bash
python -m venv testenv
testenv\Scripts\activate
pip install sonify1026 --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/
python -c "import snfy; print(snfy.SonifyEngine)"
```

If that prints the version, you are ready for the real thing.

---

## 6. Upload for real

```bash
python -m build
python -m twine check dist/*        # must say PASSED for both files
python -m twine upload dist/*
```

Expected output ends with:

```
Uploading dist/sonify1026-0.1.0-py3-none-any.whl
100% ━━━━━━━━━━ ... done
Uploading dist/sonify1026-0.1.0.tar.gz
100% ━━━━━━━━━━━━ ... done

View at:
https://pypi.org/project/sonify1026/
```

Your package is now public and installable worldwide:

```bash
pip install sonify1026
```

(The PyPI page can take a few minutes to render the first time.)

---

## 7. Verify the release

```bash
pip uninstall -y sonify1026
pip install sonify1026
python -c "import snfy; e=snfy.SonifyEngine(); print(e.synthesize('Hello!').duration)"
```

Also click through your live page at
<https://pypi.org/project/sonify1026/> and check:

- README renders correctly (this is the front page of your project)
- Both `sonify1026 "text"` and `snfy "text"` CLI commands appear under "Console script"
- Your author email is shown
- Project links point to your GitHub repo

---

## 8. Releasing future versions

You cannot overwrite a released version, so bump the number **in both files**:

| File | Field |
|---|---|
| `pyproject.toml` | `version = "0.1.0"` → `"0.1.1"` |
| `snfy/__init__.py` | `__version__ = "0.1.0"` → `"0.1.1"` |

Then rebuild and upload:

```bash
Remove-Item dist -Recurse -Force
python -m build
python -m twine check dist/*
python -m twine upload dist/*
```

Use semantic versioning: `0.1.0` → `0.1.1` (fixes), `0.2.0` (new features),
`1.0.0` (stable API).

---

## 9. Automated releases (GitHub Actions)

Two workflows publish on **Release published** (or manually via
`Run workflow`):

| Workflow | Publishes |
|---|---|
| `.github/workflows/publish-sonify1026.yml` | `sonify1026` (repo root) |
| `.github/workflows/publish-snfy.yml` | `snfy` alias (`snfy-meta/`) |

So the flow becomes: bump the version(s) → commit → create a GitHub Release for
the new tag → the workflow builds, `twine check`s and uploads to PyPI.

The alias package has its **own** version and dependency pin, so bump
`snfy-meta/pyproject.toml` (`version`, and the compatible pin
`sonify1026~=X.Y.0`) too when you want the alias to pull a newer engine. The
`~=` pin keeps the alias from silently pulling an incompatible future release.

### Authentication

The workflows support either method:

1. **API token (already configured).** A repository secret named
   `PYPI_API_TOKEN` holds a PyPI API token; the workflow passes it to
   `pypa/gh-action-pypi-publish`. Rotate it in **Settings → Secrets and
   variables → Actions**.
2. **Trusted Publishing (recommended, no secret).** On PyPI, for each project
   open **Manage → Publishing → Add a new pending publisher** with:

   | Field | Value |
   |---|---|
   | PyPI project name | `sonify1026` (and separately `snfy`) |
   | Owner | `naitik0409` |
   | Repository name | `sonify1026` |
   | Workflow name | `publish-sonify1026.yml` (and `publish-snfy.yml`) |
   | Environment name | `pypi` |

   Once added you can delete the `PYPI_API_TOKEN` secret; the `id-token: write`
   permission in the workflow makes the action use OIDC automatically when no
   password is supplied.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Invalid or non-existent authentication information` | Username must be `__token__`, password must be the `pypi-…` token (not your account password) |
| `You must be verified to upload` | Click the verification link PyPI emailed you |
| `File already exists` / `400 Client Error` | That version is already published. Bump the version number in both files |
| `Invalid distribution: Unknown distribution format` | Run `python -m build`, not `python setup.py` |
| `twine: command not found` | Use `python -m twine` |
| `403 Forbidden` | The token's scope doesn't cover this project; create a project-scoped token |
| Two-factor enabled on the account | Use a token, or pass `--username __token__ --password <token>` |

---

## Optional: source code on GitHub

Keeping the source public builds trust and lets people install from git.
The repo already lives at `github.com/naitik0409/sonify1026`:

```bash
git init
git add .
git commit -m "Sonify 1026 v0.2.0"
git branch -M main
git remote add origin https://github.com/naitik0409/sonify1026.git
git push -u origin main
```

Anyone can then use it without PyPI:

```bash
pip install git+https://github.com/naitik0409/sonify1026.git
```

Add a release tag so versions are traceable:

```bash
git tag v0.2.0
git push --tags
```