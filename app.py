
from flask import Flask, jsonify, request, Response
from ml_news_core import similarNews
import json
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
@app.route("/")
def index():
    return "Machine Learning API for Balanced News Summary" # Just updating my code

@app.route("/ml-related-news",methods=["GET"])
def api():
    args = request.args
    attempts = 0
    if "url" in args:
        while attempts < 5:
            try:
                # print(args["url"])
                # print("working")
                result_dict = similarNews(args["url"])
                
                result = jsonify(result_dict)
                # return Response(json.dumps(result_dict,indent=2), mimetype="application/json")
                # return result_dict
                break
            except Exception as e:
                result = jsonify({"Error":str(e)})
                print(e)
                print("retrying")
                attempts+=1
        return result

import os
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, port=port)