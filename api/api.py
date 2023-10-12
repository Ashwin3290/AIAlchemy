from flask import Flask, request, jsonify,url_for,send_file
from pymongo import MongoClient
from flask_restful import Api,Resource,reqparse
import pickle 
import numpy as np
import warnings
import time
import os
import json
from bson import Binary,ObjectId
import gridfs
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename
import io
import random

app = Flask((__name__))
api=Api(app) 
client = MongoClient("mongodb://localhost:27017")  
db = client["MLAlchemy"]  
dataset = db["dataset"]
user_data=db["user_data"]
model=db["model_data"]

ALLOWED_EXTENSIONS = {'xlsx', 'csv'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB

#student side api
student_roll =reqparse.RequestParser()
student_roll.add_argument("roll",type=int,help="none")
student_name=reqparse.RequestParser()
student_name.add_argument("name",type=str,help="Send student name")
class roll(Resource):
    def post(self):
        args=student_roll.parse_args()
        return {"roll":args["roll"]}

api.add_resource(roll,"/roll")

class name(Resource):
    def post(self):
        args=student_name.parse_args()
        processed=args["name"].split("-")
        return jsonify(processed)

api.add_resource(name,"/name")
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class generate_session_id(Resource):
    def get(self):
        for i in range(1000):
            sid=random.randint(1000,9999)
            if not sid in user_data.distinct("session_id"):
                return {"status":True,"session_id":sid}
        return {"status":False,"session_id":None}

api.add_resource(generate_session_id,"/session_id_gen")

class file_upload(Resource):
    def post(self):
        try:
            session_id = request.form['session_id']
            password = request.form['password']
            option=request.form["option"]
            uploaded_file = request.files['file']
            file_data = uploaded_file.read()
            if uploaded_file and allowed_file(uploaded_file.filename):
                if len(file_data) > MAX_CONTENT_LENGTH:
                    return jsonify({
                        "status": False,
                        "message": "File size exceeds the maximum allowed size (16MB)"
                    })

                filename = secure_filename(uploaded_file.filename)
                content_type = uploaded_file.content_type
                uploaded_file.seek(0)                
                result = dataset.insert_one({"file_data": file_data, "filename": filename, "content_type": content_type, "session_id": session_id, "password": password})
                file_id = str(result.inserted_id)
                file_url = url_for("get_report", file_id=file_id, _external=True)

                data = {
                    "session_id": session_id,
                    "password": password,
                    "option":option,
                    "file_id": file_id,
                    "dataset_url": file_url,
                    "model_url":None,
                    "model_id":None
                }

                user_data.insert_one(data)

                return jsonify({
                    "status": True,
                    "message": "File uploaded",
                    "file_id": file_id,
                    "file_url": file_url  
                })
            else:
                return jsonify({
                    "status": False,
                    "message": f"Invalid file format. Allowed formats:{' '.join(ALLOWED_EXTENSIONS)}"
                })
        except Exception as e:
            return jsonify({
                "status": False,
                "message": str(e)
            })


if __name__ == '__main__':
    app.run(debug=True)