# Recruiter Intelligence

AI-powered recruiter intelligence platform for:

* Job Description Analysis
* Candidate Retrieval
* Candidate Ranking
* Behavioral Intelligence
* Risk Assessment
* Explainable Scoring

## Project Structure

```text
recruiter-intelligence/
├── frontend/
├── backend/
├── data/
├── notebooks/
├── evaluation/
├── docs/
└── tests/
```

## Backend Setup

### Prerequisites

* Python 3.12+
* pip

### 1. Navigate to the backend directory

```bash
cd backend
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

#### Linux/macOS

```bash
source .venv/bin/activate
```

#### Windows

```bash
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the FastAPI server

```bash
uvicorn app.main:app --reload
```

The backend will be available at:

```text
http://127.0.0.1:8000
```

### API Documentation

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

ReDoc:

```text
http://127.0.0.1:8000/redoc
```

## Frontend Setup

### Prerequisites

* Node.js 20+
* npm

### 1. Navigate to the frontend directory

```bash
cd frontend
```

### 2. Install dependencies

```bash
npm install
```

### 3. Start the development server

```bash
npm run dev
```

The frontend will be available at:

```text
http://localhost:5173
```

## Quick Start

Open two terminals.

### Terminal 1 - Backend

```bash
cd backend

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

uvicorn app.main:app --reload
```

### Terminal 2 - Frontend

```bash
cd frontend

npm install

npm run dev
```

## Tech Stack

### Frontend

* React
* Vite
* TypeScript
* TailwindCSS
* shadcn/ui
* Recharts
* TanStack Table
* Axios

### Backend

* Python 3.12
* FastAPI
* Pydantic
* AsyncIO

## License

MIT

