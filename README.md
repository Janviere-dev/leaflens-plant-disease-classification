# LeafLens Plant Disease Classification and MLOps

LeafLens is an end-to-end plant disease classification application. A
user uploads a leaf image through the Streamlit interface, and a
TensorFlow/Keras image-classification model predicts the corresponding
plant disease class. The project includes a FastAPI inference service, a
Docker Compose deployment, model evaluation reports, and a Locust-based
load-testing setup.

## 1. Demo video

**YouTube demonstration:** `https://youtu.be/LT5_nvY3BaE`

## 2. Application URLs

### Public AWS deployment

  Service                         URL
  ------------------------------- -----------------------------------------
  LeafLens web application        https://16.171.237.246
  FastAPI Swagger documentation   https://16.171.237.246/api/docs
  FastAPI OpenAPI schema          https://16.171.237.246/api/openapi.json

### Local Docker deployment

These addresses are available on the computer running Docker. They are
not public internet URLs.

  Service                         URL
  ------------------------------- ------------------------------
  LeafLens web application        http://localhost:8502
  FastAPI service                 http://localhost:8001
  FastAPI Swagger documentation   http://localhost:8001/docs
  API health check                http://localhost:8001/health

The Docker Compose configuration binds the local service ports to
`127.0.0.1` for local-only access. In the production deployment, Nginx
provides the public HTTPS entry point and proxies requests to the
application and API.

## 3. Project features

-   **Image prediction:** submit a leaf image and receive a predicted
    plant disease class.
-   **38-class classification:** the model predicts among 38
    PlantVillage plant/disease categories.
-   **Web interface:** Streamlit pages for prediction, insights, and
    retraining workflows.
-   **REST API:** FastAPI endpoint for image inference, with interactive
    Swagger documentation.
-   **Model evaluation:** classification report, confusion matrix,
    per-class metrics, and production evaluation output.
-   **Retraining workflow:** supports the project's candidate-model
    retraining and review/approval process.
-   **Containerization:** Dockerfiles and Docker Compose services for
    repeatable deployment.
-   **Load testing:** Locust test client and an isolated HAProxy/API
    Compose stack for controlled request simulation.

## 4. Technology stack

-   Python
-   TensorFlow / Keras
-   MobileNetV2 transfer learning
-   PlantVillage image dataset
-   FastAPI and Uvicorn
-   Streamlit
-   Docker and Docker Compose
-   HAProxy
-   Locust
-   scikit-learn, NumPy, Pillow, and Hugging Face Datasets

## 5. Model and evaluation

The project uses a MobileNetV2-based image classifier with 38 output
classes. Input images are converted to RGB, resized to 224 × 224, and
preprocessed using the MobileNetV2 preprocessing function.

### Notebook evaluation

The notebook evaluation reported the following results on 10,464 test
samples:

  Metric                Result
  ------------------- --------
  Accuracy              95.84%
  Loss                  0.1664
  Macro precision       94.43%
  Macro recall          95.13%
  Macro F1-score        94.65%
  Weighted F1-score     95.80%

### Production evaluation report

The separate production evaluation report
(`reports/production_evaluation.json`) covers 10,709 test images. It
reports:

  Metric                             Result
  ------------------------ ----------------
  Accuracy                           96.11%
  Macro precision                    94.80%
  Macro recall                       95.56%
  Macro F1-score                     95.06%
  Weighted F1-score                  96.06%
  Average inference time     23.23 ms/image

**Important:** The notebook and production evaluation were run against
different test-set counts. Their results are shown separately and should
not be presented as measurements from the same evaluation run.

Detailed evaluation outputs are available in: -
`reports/classification_report.json` - `reports/confusion_matrix.csv` -
`reports/per_class_metrics.csv` - `reports/production_evaluation.json`

## 6. Repository structure

``` text

leaflens-plant-disease-classification/
├── app/
│   ├── app.py
│   └── pages/
│       ├── __init__.py
│       ├── insights.py
│       ├── prediction.py
│       └── retraining.py
├── data/
│   ├── cleaned_data/
│   │   ├── plantvillage_prepared/
│   │   │   ├── dataset_dict.json
│   │   │   ├── metadata.json
│   │   │   ├── test/
│   │   │   ├── train/
│   │   │   └── validation/
│   │   └── plantvillage_prepared.zip
│   ├── .gitkeep
│   ├── leaflens.db
│   ├── locust_sample.jpg
│   └── plantvillage/
│       ├── dataset_dict.json
│       ├── duplicate_removal_records.csv
│       ├── test/
│       └── train/
├── models/
│   ├── .gitkeep
│   ├── leaflens-trained-models/
│   │   └── leaflens-trained-models/
│   │       ├── best_mobilenetv2_class_weighted.keras
│   │       ├── best_mobilenetv2_finetuned.keras
│   │       ├── best_mobilenetv2.keras
│   │       ├── class_names.json
│   │       ├── leaflens_final_model.keras
│   │       ├── preprocessing_config.json
│   │       └── training_history*.csv
│   └── retraining/
│       └── leaflens_retrained_candidate.keras
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   └── 02-model-training.ipynb
├── reports/
│   ├── classification_report.json
│   ├── confusion_matrix.csv
│   ├── per_class_metrics.csv
│   └── production_evaluation.json
├── scripts/
│   └── evaluate_production.py
├── src/
│   ├── api_client.py
│   ├── api.py
│   ├── database.py
│   ├── disease_guidance.py
│   ├── model_loader.py
│   ├── prediction.py
│   ├── preprocessing.py
│   └── retraining.py
├── tests/
│   ├── __init__.py
│   ├── test_database.py
│   ├── test_disease_guidance.py
│   ├── test_model_loader.py
│   ├── test_prediction.py
│   ├── test_preprocessing.py
│   └── test_retraining.py
├── compose.locust.yaml
├── compose.yaml
├── Dockerfile.api
├── Dockerfile.streamlit
├── .dockerignore
├── .gitignore
├── haproxy.locust.cfg
├── locustfile.py
├── main.py
├── pyproject.toml
├── requirements.txt
├── README.md
├── uv.lock
└── project_structure.txt

```

Some generated or large assets (including the full dataset and trained
model files) are intentionally excluded from Git. They must be supplied
separately before running inference or evaluation.

## 7. Setup and run locally

### Prerequisites

-   Git
-   Docker Engine
-   Docker Compose plugin
-   Python 3.13 or a compatible environment
-   `uv` for the non-Docker Python workflow

### 7.1 Clone the repository

``` bash
git clone https://github.com/Janviere-dev/leaflens-plant-disease-classification.git
cd leaflens-plant-disease-classification
```

### 7.2 Provide the model artifacts

The trained model and its supporting metadata are not stored in this Git
repository. Obtain the project model artifact from google drive: https://drive.google.com/drive/folders/1_TyBiz4jdtag6pswxpfj3hjlJxVKMBzQ?usp=sharing  and place the files in the directory
expected by the model loader:

``` text
models/leaflens-trained-models/leaflens-trained-models/
├── leaflens_final_model.keras
├── class_names.json
└── preprocessing_config.json
```

Keep the filenames and directory structure consistent with
`src/model_loader.py`.

### 7.3 Provide the dataset (for evaluation and dataset-dependent workflows)

The PlantVillage dataset used by the project is stored locally in
Hugging Face Datasets format at:

``` text
data/plantvillage/
```

Obtain the dataset from Google Drive  https://drive.google.com/drive/folders/1_TyBiz4jdtag6pswxpfj3hjlJxVKMBzQ?usp=sharing before running notebook
evaluation or scripts that load the saved dataset. The full dataset is
excluded from Git because of its size.

### 7.4 Start the Dockerized application

From the repository root, run:

``` bash
docker compose up -d --build
```

Check the containers:

``` bash
docker compose ps
```

View logs:

``` bash
docker compose logs -f api
docker compose logs -f streamlit
```

Open the application:

-   Streamlit: http://localhost:8502
-   API documentation: http://localhost:8001/docs

Check API health:

``` bash
curl http://localhost:8001/health
```

Stop the application:

``` bash
docker compose down
```

### 7.5 Run the Python environment without Docker (optional)

Install `uv`, then run:

``` bash
uv sync
```

Start the API from the repository root:

``` bash
uv run uvicorn src.api:app --host 127.0.0.1 --port 8001
```

In another terminal, start Streamlit using the project's configured app
entry point:

``` bash
uv run streamlit run app/app.py --server.port 8502
```

The model artifacts must be present before starting inference.

## 8. API usage

The prediction endpoint accepts an image as multipart form data.

### Health check

``` bash
curl http://localhost:8001/health
```

### Predict an image

``` bash
curl -X POST \
  -F "file=@path/to/leaf-image.jpg" \
  http://localhost:8001/predict
```

The response contains the uploaded filename and the model prediction.
For the deployed API, use:

``` bash
curl -X POST \
  -F "file=@path/to/leaf-image.jpg" \
  https://16.171.237.246/predict
```

The public API is also documented at https://16.171.237.246/api/docs.

## 9. Flood request simulation (Locust)

The repository includes: - `locustfile.py` --- sends repeated multipart
image-prediction requests to `/predict`. - `compose.locust.yaml` ---
starts a separate API instance and load-balancer stack for testing. -
`haproxy.locust.cfg` --- distributes requests across the API replicas.

The load-test stack is deliberately separate from the main application
Compose project. Do not use the load-test Compose file to replace or
shut down the production application.

### 9.1 Start the isolated load-test stack

On the machine where Docker is running, from the repository root:

``` bash
docker compose -p leaflens-loadtest -f compose.locust.yaml up -d --build --scale api=1
```

Check its services:

``` bash
docker compose -p leaflens-loadtest -f compose.locust.yaml ps
```

The load-test endpoint is exposed on localhost port `8003`.

### 9.2 Run Locust

Install Locust in the local project environment if it is not already
installed:

``` bash
uv run --with locust locust -f locustfile.py --host http://127.0.0.1:8003
```

Open the Locust web interface at:

``` text
http://localhost:8089
```

Set the number of simulated users, spawn rate, and test duration. Run
controlled tests with different API replica counts (for example, 1, 2,
and 3), changing only the replica count between runs. Allow the API
replicas to become healthy before starting each run.

For a remote AWS test, establish an SSH tunnel to the instance's
localhost port 8003, then run Locust locally against
`http://127.0.0.1:8003`. Keep the load-test port private; do not expose
it publicly or open it in the AWS security group.

### 9.3 Flood request simulation results

**Enter the actual Locust measurements here after running the tests. Do
not submit estimated or invented results.**

  ------------------------------------------------------------------------------------------------------------
         API   Simulated   Spawn   Duration   Requests   Failures   Requests/sec    Median       P95   Average
    replicas       users    rate                                                   latency   latency   latency
  ---------- ----------- ------- ---------- ---------- ---------- -------------- --------- --------- ---------
           1     TO FILL TO FILL    TO FILL    TO FILL    TO FILL        TO FILL   TO FILL   TO FILL   TO FILL

           2     TO FILL TO FILL    TO FILL    TO FILL    TO FILL        TO FILL   TO FILL   TO FILL   TO FILL

           3     TO FILL TO FILL    TO FILL    TO FILL    TO FILL        TO FILL   TO FILL   TO FILL   TO FILL
  ------------------------------------------------------------------------------------------------------------

In the final report, explain how throughput and latency changed as
replicas increased, whether any requests failed, and any CPU/memory
constraints observed on the test machine. Include a screenshot or
exported Locust report as supporting evidence.

Stop only the isolated load-test stack when testing is complete:

``` bash
docker compose -p leaflens-loadtest -f compose.locust.yaml down
```

## 10. Retraining and model governance

The application includes a retraining workflow for preparing a candidate
model. Demonstrate the workflow in the UI and explain the distinction
between creating a candidate and approving/deploying a model. A newly
trained candidate should not automatically replace the production model
without evaluation and approval.

For a presentation, show the retraining steps and the status or output
produced by the workflow. Explain how evaluation results are used to
decide whether a candidate is suitable for deployment.

## 11. Presentation checklist

For a clear, user-friendly demonstration, keep the camera on and narrate
the steps:

1.  **Introduction (30--45 seconds):** introduce LeafLens, the
    plant-disease problem, and the purpose of the project.
2.  **Open the live application (30 seconds):** show the public URL and
    briefly identify the main pages.
3.  **Single-image prediction (1--2 minutes):** upload a leaf image, run
    prediction, and explain the predicted class and displayed result.
4.  **Bulk workflow (1 minute):** demonstrate the multi-image upload
    workflow if enabled, and explain how the application handles
    multiple images.
5.  **Insights and evaluation (1 minute):** show dataset/model insights
    and explain accuracy, precision, recall, F1-score, and the confusion
    matrix in plain language.
6.  **Retraining workflow (1--2 minutes):** show how a retraining run is
    initiated, what candidate output is produced, and how
    review/approval is handled. Explain that a candidate is evaluated
    before production use.
7.  **API demonstration (45 seconds):** open Swagger at `/api/docs`,
    show the health endpoint and a prediction request.
8.  **Docker and load testing (1--2 minutes):** explain the
    containerized services, show Locust, and discuss the measured
    results for different replica counts.
9.  **Conclusion (20--30 seconds):** summarize what works, the
    evaluation results, and the next improvement.

Before recording: - Test the public application and API documentation. -
Prepare a few clear leaf images for prediction. - Make sure the model
and dataset assets required by the demonstrated workflows are
available. - Run the load tests and replace every `TO FILL` value in
this README with actual measurements. - Replace the YouTube placeholder
with the published video URL. - Keep the camera on and ensure the
screen, voice, and results are readable.

## 12. Limitations and responsible use

LeafLens is an image-classification demonstration, not a substitute for
expert agronomic diagnosis. Image quality, lighting, plant variety,
disease stage, and differences between training and real-world images
can affect predictions. A prediction should be treated as decision
support and verified when the result affects crop treatment or financial
decisions.
