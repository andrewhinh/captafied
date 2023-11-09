"""AWS Lambda function serving inference predictions."""
import json

from inference.inference import Pipeline
import pandas as pd


model = Pipeline()


def handler(event, _context):
    """Provide main prediction API."""
    print("INFO loading input")
    table = _load_table(event)
    if table is None:
        return {"statusCode": 400, "message": "table not found in event"}
    requests = _load_requests(event)
    if requests is None:
        return {"statusCode": 400, "message": "requests not found in event"}
    prev_answers = _load_prev_answers(event)
    print("INFO input loaded")
    print("INFO starting inference")
    pred = model.predict(table, requests, prev_answers)
    print("INFO inference complete")
    print("METRIC num_pred_answers {}".format(sum(len(output) > 0 for output in pred)))
    print("INFO pred {}".format(pred))
    return {"pred": pred}


def _from_string(event):
    if isinstance(event, str):
        return json.loads(event)
    else:
        return event


def get_event(event):
    event = _from_string(event)
    return _from_string(event.get("body", event))


def _load_table(event):
    event = get_event(event)
    table = event.get("table")
    if table is not None:
        print("INFO reading table from event")
        return pd.DataFrame(table)
    else:
        return None


def _load_requests(event):
    event = get_event(event)
    requests = event.get("requests")
    if requests is not None:
        print("INFO reading requests from event")
        return requests
    else:
        return None


def _load_prev_answers(event):
    event = get_event(event)
    prev_answers = event.get("prev_answers")
    if prev_answers is not None:
        print("INFO reading prev_answers from event")
        return prev_answers
    else:
        return None
