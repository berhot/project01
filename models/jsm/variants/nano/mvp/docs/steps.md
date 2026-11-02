# Environment setup

Create and activate the shared development environment from the company repository root:

```bash
python3 -m venv .venv
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

On Windows:

```powershell
.venv\Scripts\activate
```

Install this model's dependencies and enter its working directory:

```bash
python -m pip install -r models/jsm/variants/nano/mvp/requirements.txt
cd models/jsm/variants/nano/mvp
```

Continue with the [MVP guide](../README.md).
