Tamr Vision: minimum package to run the complete system end to end
====================================================================
Course: Computer Vision Systems Development (SDA-AIE-212), SDAIA Academy

What is in here
  run_demo.py               the whole system in one script (5 steps, prints DONE at the end)
  Tamr_Vision_Colab.ipynb   open this in Google Colab and run the cells top to bottom
  tamr_vision.py            the course library (models, contract, harness, export helpers)
  make_dates_qc.py          generates the dataset (run automatically by run_demo.py)
  config/grading_rules.json the plant's grading rules; the ONLY place thresholds live
  checkpoints/              trained models: grade_resnet18.pt, defect_y11n.pt, defect_y11n_seg.pt

What run_demo.py produces
  data/dates-qc/            1,200 frames with labels (generated)
  dist/*.onnx               classifier FP32 and INT8, detector at 256 px
  runs/decision_logs.json   one JSON line per frame: decision, reason, scores, latency, versions
  report.html               open in a browser: numbers, tables, and every frame with its decision

Run on your own computer instead of Colab
  pip install -r requirements.txt
  python run_demo.py
