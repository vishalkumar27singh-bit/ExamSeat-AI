# macOS setup (nothing is reinstalled)

Your environment list earlier showed `base`, `primeen` and `tensorflow`; `base` has no PaddleOCR, `tensorflow` does.

```bash
conda activate tensorflow
cd path/to/examseat_ai
pip list | grep -i -E "opencv|pandas|pillow|paddle|streamlit|numpy|tabulate|pytest"
```

Install ONLY what is missing from that list:

```bash
pip install streamlit tabulate            # app + evaluation report
pip install pytest                        # tests
pip install rapidocr-onnxruntime          # optional offline fallback OCR
```

First run needs internet once (PaddleOCR downloads its models). If it cannot, the app says so and uses RapidOCR.

## Run

```bash
streamlit run app.py                      # multi-photo app: upload several photos, type one number
python multi_search.py "<registration-number>" a.jpg b.jpg c.jpg  # same search from the terminal
python -m pytest tests -q                 # automated tests (about 1-2 minutes)
python evaluation.py                      # accuracy on ground_truth/*.json
python diagnose_coordinates.py "<photo>" "<registration-number>" # PaddleOCR coordinate check (please run once and keep the output)
```

Apple-silicon notes: `paddlepaddle` and `opencv-python` have arm64 wheels; if PaddleOCR fails to start, choose
"rapid" in the app's sidebar. Large photos (12 MP) take several seconds each; the app shows progress per photo.

## Optional custom ML

```bash
pip install ultralytics                   # large (installs torch); only if you annotate data - see ml/ANNOTATION_GUIDE.md
```
