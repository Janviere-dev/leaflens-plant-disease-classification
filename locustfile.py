from pathlib import Path

from locust import HttpUser, between, task


IMAGE_PATH = Path(__file__).resolve().parent / "data" / "locust_sample.jpg"


class LeafLensUser(HttpUser):
    wait_time = between(0.5, 1.5)

    def on_start(self):
        self.image_bytes = IMAGE_PATH.read_bytes()

    @task
    def predict_leaf(self):
        with self.client.post(
            "/predict",
            files={
                "file": (
                    "locust_sample.jpg",
                    self.image_bytes,
                    "image/jpeg",
                )
            },
            name="/predict",
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(
                    f"HTTP {response.status_code}: {response.text[:200]}"
                )
                return

            try:
                result = response.json()
                if "prediction" not in result:
                    response.failure("Response has no prediction field")
                else:
                    response.success()
            except ValueError:
                response.failure("Response is not valid JSON")
