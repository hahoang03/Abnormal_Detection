# Abnormal Detection

## 1. Clone project

```bash
git clone https://github.com/hahoang03/Abnormal_Detection.git
cd Abnormal_Detection
```

---

## 2. Add model files

Model files are not included in GitHub.  
Put these files inside the `BE/` folder:

```txt
BE/hcunet.pth
BE/unetplus_baseline.pth
```

Expected structure:

```txt
BE/
├── app.py
├── manipulation_generator.py
├── hcunet.pth
└── unetplus_baseline.pth
```

---

## 3. Setup Backend

```bash
cd BE
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

If `requirements.txt` does not exist, create it inside `BE/`:

```txt
fastapi
uvicorn
python-multipart
pillow
opencv-python
numpy
torch
torchvision
segmentation-models-pytorch
diffusers
transformers
accelerate
safetensors
```

---

## 4. Run Detection API

Open terminal 1:

```bash
cd Abnormal_Detection/BE
source venv/bin/activate
uvicorn app:app --reload --port 8000
```

Detection API:

```txt
http://localhost:8000
```

---

## 5. Run Generate API

Open terminal 2:

```bash
cd Abnormal_Detection/BE
source venv/bin/activate
uvicorn manipulation_generator:app --reload --port 8001
```

Generate API:

```txt
http://localhost:8001
```

---

## 6. Setup Frontend

Open terminal 3:

```bash
cd Abnormal_Detection/FE
npm install
```

Create `FE/.env.local`:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_GENERATE_API_URL=http://localhost:8001
```

---

## 7. Run Frontend

```bash
npm run dev
```

Open:

```txt
http://localhost:3000
```

---

## Run Summary

You need 3 terminals:

```bash
# Terminal 1: Detection API
cd BE
source venv/bin/activate
uvicorn app:app --reload --port 8000
```

```bash
# Terminal 2: Generate API
cd BE
source venv/bin/activate
uvicorn manipulation_generator:app --reload --port 8001
```

```bash
# Terminal 3: Frontend
cd FE
npm run dev
```

---

## Common Error

If `/api/generate` returns `404 Not Found`, make sure you are running:

```bash
uvicorn manipulation_generator:app --reload --port 8001
```

Do not run this on port `8001`:

```bash
uvicorn app:app --reload --port 8001
```
