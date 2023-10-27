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
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename
import io
import random
import pandas as pd
from pycaret.regression import RegressionExperiment
import pycaret.classification as ClassificationExperiment
from datetime import datetime

app = Flask((__name__))
api=Api(app) 
client = MongoClient("mongodb://localhost:27017")  
db = client["Mlalchemy"]  
dataset = db["dataset"]
user_data=db["user_data"]
model=db["model_data"]


credential=reqparse.RequestParser()
credential.add_argument("session_id",type=str,help="session_id is required",required=True)
credential.add_argument("password",type=str,help="password is required",required=True)
credential.add_argument("target",type=str,help="target is required",required=True)

ALLOWED_EXTENSIONS = {'xlsx', 'csv'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def in_user(session_id,password):
    return user_data.find_one({"session_id":session_id,"password":password})

def get_session_details(Resource):
    return user_data.find_one({"session_id":session_id,"password":password})

def check_validity(session_id,password):
    if user_data.find_one({"session_id":session_id,"password":password}):
        include={"file_id":1}
        data=user_data.find_one({"session_id":session_id,"password":password},include)
        file_data=dataset.find({"_id":ObjectId(data["file_id"])},{"file_data":1,"content_type":1})
        file_data=list(file_data)[0]
        if file_data["content_type"]=="text/csv":
            df=pd.read_csv(io.BytesIO(file_data["file_data"]))
        elif file_data["content_type"]=="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            df=pd.read_excel(io.BytesIO(file_data["file_data"]))
        else:
            return {"status":False,"message":"Invalid file format"}
        
        return df
    
    return {"status":False,"message":"Invalid session_id or password"}

class generate_session_id(Resource):
    def get(self):
        for i in range(1000):
            sid=random.randint(1000,9999)
            if not sid in user_data.distinct("session_id"):
                chars=['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', 'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'z', 'x', 'c', 'v', 'b', 'n', 'm','1','2','3','4','5','6','7','8','9','0']
                password="".join(random.choices(chars, k=6))
                data = {
                    "time_stamp":datetime.now(),
                    "session_id": str(sid),
                    "password": password,
                    "option":None,
                    "file_id": None,
                    "content_type": None,
                    "dataset_url": None,
                    "model_url":None,
                    "model_id":None
                }
                user_data.insert_one(data)
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
                time_stamp=datetime.now()
                result = dataset.insert_one({"file_data": file_data,"time_stamp":time_stamp, "filename": filename, "content_type": content_type, "session_id": session_id, "password": password})
                file_id = str(result.inserted_id)
                file_url = url_for("get_dataset", file_id=file_id, _external=True)

                data = {
                    "time_stamp":time_stamp,
                    "session_id": session_id,
                    "password": password,
                    "option":option,
                    "file_id": file_id,
                    "content_type": content_type,
                    "dataset_url": file_url,
                    "model_url":None,
                    "model_id":None
                }

                user_data.delete_one({"session_id":session_id,"password":password})
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


@app.route("/get_model/<file_id>")
def get_model(model_id):
    file_data = model.find_one({"_id":ObjectId(model_id)},{"model":1})
    data=user_data.find_one({"model_id":model_id})
    if file_data:
        return send_file(
            io.BytesIO(file_data["model"]),
            mimetype="application/octet-stream",
            as_attachment=True,
            download_name="model.pkl"
        )
    else:
        return jsonify({
            "status": False,
            "message": "File not found"
        })

@app.route("/get_dataset/<file_id>")
def get_dataset(file_id):
    file_data = dataset.find_one({"_id":ObjectId(file_id)},{"file_data":1,"filename":1})
    data=user_data.find_one({"file_id":file_id})
   
    if file_data:
        return send_file(
            io.BytesIO(file_data["file_data"]),
            mimetype="application/octet-stream",
            as_attachment=True,
            download_name=file_data["filename"]
        )
    else:
        return jsonify({
            "status": False,
            "message": "File not found"
        })


class regression(Resource):
    def get(self):
        args=credential.parse_args()
        session_id=args["session_id"]
        password=args["password"]
        choosen_target=args["target"]
        rgr=RegressionExperiment()
        print("Experimnet")
        validity=check_validity(session_id,password)
        if isinstance(validity,pd.DataFrame):
            df=validity
            print("validity")
        else:
            return validity
        rgr.setup(df, target=choosen_target)
        best_model = rgr.compare_models()
        compare_df = rgr.pull()
        print("compare_df")
        rgr.finalize_model(best_model)
        pipeline=pickle.dumps(rgr.save_model(best_model, model_name='best_model'))
        os.remove("best_model.pkl")
        compare_df = pickle.dumps(compare_df)
        timestamp=datetime.now()
        model_id=model.insert_one({"time_stamp":timestamp,"model":pipeline,"compare_df":compare_df})
        print("model_id")
        model_url=url_for("get_model",file_id=model_id,_external=True)
        user_data.update_one({"session_id":session_id,"password":password},{"$set":{"model_url":model_url,"model_id":str(model_id),"time_stamp":timestamp}})
        return {"status":True,"message":"Model created successfully"}
api.add_resource(regression,"/regression")


class classification(Resource):
    def get(self):
        args=credential.parse_args()
        session_id=args["session_id"]
        password=args["password"]
        choosen_target=args["target"]
        clf=ClassificationExperiment
        print("Experimnet")
        validity=check_validity(session_id,password)
        if isinstance(validity,pd.DataFrame):
            df=validity
            print("validity")
        else:
            return validity
        clf.setup(df, target=choosen_target)
        best_model = clf.compare_models()
        compare_df = clf.pull()
        print("compare_df")
        clf.finalize_model(best_model)
        pipeline=pickle.dumps(clf.save_model(best_model, model_name='best_model'))
        os.remove("best_model.pkl")
        compare_df = pickle.dumps(compare_df)
        timestamp=datetime.now()
        model_id=model.insert_one({"time_stamp":timestamp,"model":pipeline,"compare_df":compare_df})
        print("model_id")
        model_url=url_for("get_model",file_id=model_id,_external=True)
        user_data.update_one({"session_id":session_id,"password":password},{"$set":{"model_url":model_url,"model_id":str(model_id),"time_stamp":timestamp}})
        return {"status":True,"message":"Model created successfully"}
api.add_resource(classification,"/classification")

class get_plots(Resource):
    def get(self):
        args=credential.parse_args()
        session_id=args["session_id"]
        password=args["password"]
        model_id=user_data.find_one({"session_id":session_id,"password":password},{"model_id":1})
        model_data=model.find_one({"_id":ObjectId(model_id)})
        pipeline=pickle.loads(model_data["model"])
        feature_importance = plot_model(pipeline, plot='feature')
        plt.tight_layout()
        image_data = BytesIO()
        plt.savefig(image_data, format='png')
        image_data.seek(0)
        response = Response(image_data.read(), content_type='image/png')
        return response 
api.add_resource(get_plots,"/plots")

if __name__ == '__main__':
    app.run(debug=True)