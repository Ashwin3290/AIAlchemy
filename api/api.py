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
import pycaret.regression as rgr

app = Flask((__name__))
api=Api(app) 
client = MongoClient("mongodb://localhost:27017")  
db = client["MLAlchemy"]  
dataset = db["dataset"]
user_data=db["user_data"]
model=db["model_data"]


credential=reqparse.RequestParser()
credential.add_argument("session_id",type=int,help="session_id is required",required=True)
credential.add_argument("password",type=str,help="password is required",required=True)

ALLOWED_EXTENSIONS = {'xlsx', 'csv'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class generate_session_id(Resource):
    def get(self):
        for i in range(1000):
            sid=random.randint(1000,9999)
            if not sid in user_data.distinct("session_id"):
                password=random.choices(['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', 'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'z', 'x', 'c', 'v', 'b', 'n', 'm','1','2','3','4','5','6','7','8','9','0'], k=6)
                data = {
                    "session_id": session_id,
                    "password": password,
                    "option":None,
                    "file_id": None,
                    "dataset_url": None,
                    "model_url":None,
                    "model_id":None
                }

                return {"status":True,"session_id":sid,"password":password}

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
                    "content_type": content_type,
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

api.add_resource(file_upload,"/upload")

class regression(Resource):
    def get(self):
        args=credential.parse_args()
        session_id=args["session_id"]
        password=args["password"]
        if user_data.find_one({"session_id":session_id,"password":password}):
            include={"file_id":1,"content_type":1}
            data=user_data.find_one({"session_id":session_id,"password":password},include)
            if data["content_type"]=="application/csv":
                df=pd.read_csv(io.bytesIO(data["file_data"]))
            elif data["content_type"]=="application/xlsx":
                df=pd.read_excel(io.bytesIO(data["file_data"]))
            rgr.setup(df, target=chosen_target, silent=True)
            setup_df = pull()
            best_model = compare_models()
            compare_df = pull()
            save_model(best_model, 'best_model')

if __name__ == '__main__':
    app.run(debug=True)