# Abnormal Detection Demo

## Setup Backend

Put model weights inside BE folder:

- hcunet.pth
- unetplus_baseline.pth

Then run:

```bash
cd BE
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
