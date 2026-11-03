# Public deployment

## Platform

Use [Streamlit Community Cloud](https://share.streamlit.io/). ExamSeat AI is a single
Streamlit application in `app.py`; it does not have a separate frontend/API, and it does
not require a database or deployment secrets. The public GitHub repository is the source:

`https://github.com/vishalkumar27singh-bit/ExamSeat-AI`

## Deploy

1. Sign in to Streamlit Community Cloud with the GitHub account that has access to the repository.
2. Select **Create app** → **Yup, I have an app**.
3. Choose repository `vishalkumar27singh-bit/ExamSeat-AI`, branch `main`, and entrypoint `app.py`.
4. In **Advanced settings**, choose Python 3.12. No secrets need to be entered.
5. Optionally select an available custom subdomain, then select **Deploy**.
6. Wait for the app health indicator to show running and open the assigned `https://…streamlit.app` URL.

The first launch installs `requirements.txt` and initializes the RapidOCR/ONNX runtime.
`packages.txt` installs the system OpenGL library required by the `opencv-python` wheel
on Streamlit Community Cloud.
The hosted app offers RapidOCR. PaddleOCR is an optional local backend and is deliberately
not installed on the free hosted instance to reduce build size and memory requirements.

## Resource and privacy behavior

- Up to 5 images per search; 10 MB per image; 30 MB total; at most 20 megapixels per image.
- Images and entered registration numbers are sent to the Streamlit host for processing.
  The application does not write uploads/results to a database or persistent files; image
  data and OCR results are kept in per-session memory while the session is active.
- The sidebar's **Clear this session's photos and results** action clears the app's cached
  analysis/results and uploaded-file widget state.
- Do not upload student records or photos unless you are authorized to process them.
  This is a public, unauthenticated application: anyone with its URL can use it. The
  hosting provider's own operational policies still apply.
- There is no database, user account, or durable history. Restarting the app/session loses
  in-memory state. Free-host resource limits and cold starts can affect processing time.

## Smoke-test after deployment

1. Confirm the page loads over HTTPS and shows the ExamSeat AI title and privacy note.
2. Upload an authorized test chart and search a known number.
3. Verify that the result status is shown and that room/column/row appear only for a
   verified seat; confirm the highlighted image and downloadable result.
4. Try an invalid/non-matching query and a corrupt image; confirm the app reports these
   without crashing or stopping other photos.
5. Test the upload limits and the clear-session action.

Never use an actual student's exam chart as a deployment smoke-test unless authorized.
