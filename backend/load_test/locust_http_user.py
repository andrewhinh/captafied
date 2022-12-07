import json

from locust import constant, HttpUser, task


table_url = "inference/tests/support/tables/temp.csv"


class User(HttpUser):
    """
    Simulated AWS Lambda User
    """

    wait_time = constant(1)
    headers = {"Content-type": "application/json"}
    payload = json.dumps({"table_url": table_url})

    @task
    def predict(self):
        response = self.client.post("/", data=self.payload, headers=self.headers)
        pred = response.json()["pred"]
