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

app.config['dataset'] = db["dataset"]
app.config['user_data'] = db["user_data"]
app.config['model'] = db["model_data"]

credential=reqparse.RequestParser()
credential.add_argument("session_id",type=str,help="session_id is required",required=True)
credential.add_argument("password",type=str,help="password is required",required=True)
credential.add_argument("target",type=str,help="target is required",required=True)


plot=reqparse.RequestParser()
plot.add_argument("session_id",type=str,help="session_id is required",required=True)
plot.add_argument("password",type=str,help="password is required",required=True)

app.config['credential'] = credential
app.config['plot'] = plot



ALLOWED_EXTENSIONS = {'xlsx', 'csv'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def in_user(session_id,password):
    return user_data.find_one({"session_id":session_id,"password":password})

def get_session_details(session_id,password):
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
def get_model(file_id):
    file_data = model.find_one({"_id":ObjectId(file_id)},{"model":1})
    data=user_data.find_one({"model_id":file_id})
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
        validity=check_validity(session_id,password)
        if isinstance(validity,pd.DataFrame):
            df=validity
        else:
            return validity
        rgr.setup(df, target=choosen_target)
        best_model = rgr.compare_models()
        compare_df = rgr.pull()
        rgr.finalize_model(best_model)
        pipeline=pickle.dumps(rgr.save_model(best_model, model_name='best_model'))
        os.remove("best_model.pkl")
        model_comparison = compare_df.to_dict()
        compare_df = pickle.dumps(compare_df)
        timestamp=datetime.now()
        model_id=model.insert_one({"time_stamp":timestamp,"model":pipeline,"compare_df":compare_df}).inserted_id
        model_url=url_for("get_model",file_id=model_id,_external=True)
        user_data.update_one({"session_id":session_id,"password":password},{"$set":{"model_url":model_url,"model_id":str(model_id),"time_stamp":timestamp}})
        return {"status":True,"model_url":model_url,"message":"Model created successfully","model_comparison":model_comparison}
api.add_resource(regression,"/regression")


class classification(Resource):
    def get(self):
        args=credential.parse_args()
        session_id=args["session_id"]
        password=args["password"]
        choosen_target=args["target"]
        clf=ClassificationExperiment
        validity=check_validity(session_id,password)
        if isinstance(validity,pd.DataFrame):
            df=validity
        else:
            return validity
        clf.setup(df, target=choosen_target)
        best_model = clf.compare_models()
        compare_df = clf.pull()
        clf.finalize_model(best_model)
        pipeline=pickle.dumps(clf.save_model(best_model, model_name='best_model'))
        os.remove("best_model.pkl")
        model_comparison = compare_df.to_dict()
        compare_df = pickle.dumps(compare_df)
        timestamp=datetime.now()
        model_id=model.insert_one({"time_stamp":timestamp,"model":pipeline,"compare_df":compare_df}).inserted_id
        model_url=url_for("get_model",file_id=model_id,_external=True)
        user_data.update_one({"session_id":session_id,"password":password},{"$set":{"model_url":model_url,"model_id":str(model_id),"time_stamp":timestamp}})
        return {"status":True,"model_url":model_url,"message":"Model created successfully","model_comparison":model_comparison}
api.add_resource(classification,"/classification")

class get_plots(Resource):
    def get(self):
        args=plot.parse_args()
        session_id=args["session_id"]
        password=args["password"]
        model_id=list(user_data.find_one({"session_id":session_id,"password":password},{"model_id":1}))["model_id"]
        model_data=model.find_one({"_id":ObjectId(model_id)})
        pipeline=pickle.loads(io.BytesIO(model_data["model"]))
        plt.figure(figsize=(10, 8))
        plot_model(pipeline, plot='feature')
        plt.tight_layout()

        image_data = BytesIO()
        plt.savefig(image_data, format='png')
        image_data.seek(0)

        response = Response(image_data.read(), content_type='image/png')
        return response

api.add_resource(get_plots,"/plots")


class ClusteringResource(Resource):
    def get(self):
        data=request.get_json()
        session_id=args["session_id"]
        password=args["password"]
        clusters=int(data['clusters'])

        validity=check_validity(session_id,password)
        if isinstance(validity,pd.DataFrame):
            data=validity
        else:
            return validity

        object_columns = data.select_dtypes(include=['object']).columns

        # Identify columns that might contain dates and convert them to datetime
        for column in object_columns:
            try:
                data[column] = pd.to_datetime(data[column])
            except (TypeError, ValueError):
                pass  # Ignore columns that cannot be converted timestamp

        timestamp_columns = data.select_dtypes(include=['datetime64']).columns
        if timestamp_columns.any():
            primary_timestamp_column = timestamp_columns[0]
            data.drop(columns=list(timestamp_columns[1:]), inplace=True)
            data.set_index(primary_timestamp_column, inplace=True)
            # data = data.resample('D').sum()
            data.index = pd.to_datetime(data.index)

        data = data.select_dtypes(exclude=['object'])
        data.fillna(method="ffill", inplace=True)
        data.dropna(inplace=True)

        # Outlier detection and removal using Z-score
        z_scores = stats.zscore(data.select_dtypes(include=['float64', 'int64']))
        abs_z_scores = abs(z_scores)
        filtered_entries = (abs_z_scores < 3).all(axis=1)
        data = data[filtered_entries]

        # Check and handle stationarity
        for column in data.columns:
            d = ndiffs(data[column], test='adf')
            if d > 0:
                data[column] = data[column].diff(d)

        data.fillna(method="ffill", inplace=True)
        data.dropna(inplace=True)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)

        # silhouette_scores = []
        # for i in range(2, 11):
        #     kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        #     kmeans.fit(scaled_data)
        #     labels = kmeans.labels_

        #     # Check condition 1: All clusters should have a silhouette score greater than the average score
        #     avg_score = silhouette_score(scaled_data, labels)
        #     sample_silhouette_values = silhouette_samples(scaled_data, labels)
            
        #     if all(s > avg_score for s in sample_silhouette_values):
        #         # Check condition 2: Avoid wide fluctuations in the size of clusters
        #         cluster_sizes = [np.sum(labels == j) for j in range(i)]
        #         if max(cluster_sizes) / min(cluster_sizes) < 2.0:
        #             silhouette_scores.append((i, avg_score))

        # # Find the optimal number of clusters with the highest average silhouette score
        # optimal_clusters = max(silhouette_scores, key=lambda x: x[1], default=(2, 0))[0]+1
        # print("Optimal number of clusters:", optimal_clusters)

        # # Plot silhouette scores
        # x_values, y_values = zip(*silhouette_scores)
        # plt.plot(x_values, y_values, marker='o')
        # plt.axhline(y=np.mean(y_values), color="red", linestyle="--", label="Average Silhouette Score")
        # plt.title('Silhouette Score Method')
        # plt.xlabel('Number of Clusters')
        # plt.ylabel('Silhouette Score')
        # plt.legend()
        # plt.show()

        # Perform K-means clustering with the optimal number of clusters
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_data)

        # 2. Perform K-means Clustering on PCA result
        kmeans = KMeans(n_clusters=clusters , random_state=0)  # Update the number of clusters
        data['Cluster'] = kmeans.fit_predict(pca_result)

        # 3. Visualize the Clusters in 2D PCA Space
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=data['Cluster'], palette='viridis', legend='full')
        plt.title(f'Clusters in 2D PCA Space (Number of Clusters: {clusters})')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.show()

        # Analyze the characteristics of each cluster
        cluster_stats = data.groupby('Cluster').mean()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        # Convert the BytesIO object to a base64 string
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()

        # Create a data URL from the base64 string
        image_data_url = 'data:image/png;base64,' + urllib.parse.quote(image_base64)
        return {"status": True, "cluster_plot": image_data_url,"cluster_stats":cluster_stats.to_dict()}

api.add_resource(ClusteringResource, '/clustering')

class get_cluster_plots(Resource):
    def get(self):
        args=plot.parse_args()
        session_id=args["session_id"]
        password=args["password"]
        model_id=list(user_data.find_one({"session_id":session_id,"password":password},{"model_id":1}))["model_id"]
        model_data=model.find_one({"_id":ObjectId(model_id)})
        pipeline=pickle.loads(io.BytesIO(model_data["model"]))
        plt.figure(figsize=(10, 8))
        plot_model(pipeline)
        plt.tight_layout()
        image_data = BytesIO()
        plt.savefig(image_data, format='png')
        image_data.seek(0)

        response = Response(image_data.read(), content_type='image/png')
        return response

api.add_resource(get_cluster_plots,"/cluster_plots")


if __name__ == "__main__":
    app.run(debug=True)

#pip install waitress
#waitress-serve --port=5000 api:create_app