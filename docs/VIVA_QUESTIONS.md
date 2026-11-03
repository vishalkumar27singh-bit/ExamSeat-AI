# Viva questions with short answers (based on THIS project)

**1. What problem does ExamSeat AI solve?** Finds a student's room, seat column and row from phone photos of sitting charts, across many photos.

**2. Why not just OCR the whole page and use the order of the text?** Reading order does not give the seat. Column and row come from the printed grid lines (OpenCV), not from OCR order.

**3. Why do you keep OCR, detection and seat mapping separate?** Each can be tested and replaced alone: OCR reads text, the seat mapper turns lines into A-F/row, a detector (optional) only proposes regions.

**4. What was the coordinate bug?** The highlight landed on a neighbouring number. PaddleOCR 3.x pre-processes the image by default (orientation + unwarping), so polygons are in the processed image, not the original.

**5. How did you fix it?** Switched those stages off, used only the aligned `rec_*` lists, raised an error if the processed size differs, and re-read every box.

**6. Is that fix proven on real PaddleOCR?** Not yet. Verified with a mock and with RapidOCR; `diagnose_coordinates.py` must be run on the real setup.

**7. What does "re-read verification" mean?** Crop the highlighted polygon from the original photo, OCR it alone, and require the same number.

**8. Why can a highlight at the photo edge fail verification?** The number is cut off; re-reading gives a different string. Refusing is correct.

**9. How do you handle O vs 0?** A token containing a letter that had to be changed is never an exact match; it appears only as a candidate that the user must confirm.

**10. What happens when no exact match exists?** "Registration Number Not Found" plus similar numbers; none is reported as the seat until the user confirms.

**11. How is the seat column found?** Vertical lines via ridge filter + Hough; the chain of walls around the number must give exactly 6 columns, each containing numbers.

**12. How is the row found?** Horizontal lines visible in both text-free margins of the cell; the first thinner band above a run of equal rows is the header (A-F), and row 1 is just below it.

**13. What is the ridge filter?** A second-derivative-of-Gaussian filter that responds to thin dark lines; better than plain edges for printed table lines.

**14. What is the Hough transform?** Votes in line-parameter space so collinear pixels form a peak - robust to gaps in a line.

**15. Why de-skew?** Phone photos are tilted; aligned rows/columns make the profiles sharp and the line search simpler. Coordinates are mapped back with a `Transform`.

**16. What do you do when a table wall is missing?** If the found columns are evenly spaced (5 or 7), the centred "NIST UNIVERSITY" title decides which six form the table; an inferred wall lowers confidence and is flagged. Otherwise: unverified.

**17. What is the "unique seat" check?** Two different numbers can never share a seat; if they do, neither is trusted. It caught a wrong placement during development.

**18. Why upload multiple photos?** A hall has many sheets, so a student's number may be in any photo. Each photo is processed independently.

**19. What if one photo is corrupt?** It gets an ERROR row with the reason; the other photos are still searched (tested).

**20. What if the number is in two photos?** Every match is shown; the app says whether they agree on one verified seat, include a cut-off view, or conflict. "Best match" is chosen only when verified appearances agree.

**21. How common is that?** In 14 real files, 189 of 974 distinct numbers appeared in more than one (neighbouring sheets overlap; one photo was included twice).

**22. How do you avoid running OCR twice for the same photo?** `multi_search` caches the analysis by file hash and OCR engine; a second search on the same photos is fast (tested).

**23. Did you train a YOLO model?** No. Only one annotated real photo exists. The tooling is built and smoke-tested; training on one sheet would only memorise it.

**24. Why split by photo group, not by file?** Copies of one sheet in both train and validation would leak and inflate the score.

**25. What would YOLO add?** Possibly cleaner table crops before the grid code; unproven. It must beat the grid code on unseen photos first.

**26. What accuracy do you report?** 96/96 correct on the one real photo (tuned on it), degraded copies as in the report, and no accuracy claim for the 13 other photos because they have no ground truth - only consistency checks.

**27. What is a "confident error"?** A result marked HIGH confidence that is wrong. Measured 0 on all tested conditions.

**28. What are the confidence levels?** Confidence = OCR/re-read score x grid checks. HIGH needs a verified seat and >= 0.7; MEDIUM >= 0.45; otherwise LOW.

**29. What are the main limitations?** Whole table + header row must be visible; layout-specific rules; curved/shaded photos get refused; real PaddleOCR run still pending.

**30. How would you improve it?** Annotate 30+ independent photos, evaluate a table detector, add page-curvature correction, guide the user while photographing.
