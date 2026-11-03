# macOS setup

Use Python 3.12 for parity with Streamlit Community Cloud. The runtime dependencies below
use RapidOCR by default; PaddleOCR can be installed separately as an optional local backend.

```bash
cd path/to/ExamSeat-AI
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install pytest              # optional, to run the included test
```

To use PaddleOCR locally as the preferred engine, additionally install
`paddlepaddle>=3.0` and `paddleocr>=3.0`. The app uses RapidOCR if PaddleOCR is not installed
or cannot start. The first OCR run may need internet access to obtain model assets.

## Run

```bash
streamlit run app.py                      # multi-photo app: upload several photos, type one number
python multi_search.py "<registration-number>" a.jpg b.jpg c.jpg  # same search from the terminal
python -m pytest tests/test_matching.py -q # data-independent tests included in Git
python evaluation.py                      # needs private ground_truth/ data (not in public Git)
python diagnose_coordinates.py "<photo>" "<registration-number>" # PaddleOCR coordinate check (please run once and keep the output)
```

Photo-dependent tests and evaluation data are local-only and are not included in the public repository.
The hosted app accepts up to five photos, 10 MB per photo, 30 MB total, and 20 megapixels per image.

## Optional custom ML

```bash
pip install ultralytics                   # large (installs torch); only if you annotate data - see ml/ANNOTATION_GUIDE.md
```
