#
# Download models from Google
#

import urllib
import json

import tempfile
import os
import tfjs_graph_converter.api as tfjs

base_url = "https://storage.googleapis.com/tfjs-models/savedmodel/bodypix/"

def load_model(model_type, stride, quant_bytes, multiplier):
    """
    Load model from google storage
    """
    query_url = base_url + f"{model_type}/"
    if quant_bytes == 4:
        query_url += "float/"
    else:
        query_url += f"quant{quant_bytes}/"
    if model_type == "mobilenet":
        query_url += f"{multiplier*100:03.0f}/"

    graph_url = query_url + f"model-stride{stride}.json"

    response = urllib.request.urlopen(graph_url)
    graph_data = json.loads(response.read())

    with tempfile.TemporaryDirectory() as temp_dir:
        with open(os.path.join(temp_dir, 'model.json'), 'w') as model_js:
            model_js.write(json.dumps(graph_data))
        for weight in graph_data['weightsManifest'][0]['paths']:
            weight_url = query_url + f"{weight}"
            print(weight_url)
            response = urllib.request.urlopen(weight_url)
            with open(os.path.join(temp_dir, weight), 'wb') as wf:
                wf.write(response.read())
        return tfjs.load_graph_model(temp_dir)
