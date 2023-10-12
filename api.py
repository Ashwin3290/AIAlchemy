from flask import Flask, request, jsonify
from flask_restful import Api,Resource,reqparse
from pymongo import MongoClient
from bson import Binary
import gridfs

app = Flask((__name__))
api=Api(app) 
client = MongoClient("mongodb://localhost:27017")  
db = client["MLAlchemy"]  
collection = db["data"] 

ALLOWED_EXTENSIONS = {'xlsx', 'csv'}
MAX_CONTENT_LENGTH = 30 * 1024 * 1024  # 30MB

app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
fs=gridfs(GridFS(db),collection="data")


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

class file_upload(Resource):
    def post(self):
        try:
            userid = request.form['userid']
            time = request.form['time']
            location = request.form['location']
            file_type = request.form['type']
            uploaded_file = request.files['file']
            file_data = uploaded_file.read()
            if uploaded_file and allowed_file(uploaded_file.filename):
                if len(file_data) > MAX_CONTENT_LENGTH:
                    return jsonify({
                        "status": False,
                        "message": "File size exceeds the maximum allowed size (5MB)"
                    })

                filename = secure_filename(uploaded_file.filename)
                content_type = uploaded_file.content_type
                uploaded_file.seek(0)                
                result = report.insert_one({"file_data": file_data})
                file_id = str(result.inserted_id)
                file_url = url_for("get_report", file_id=file_id, _external=True)

                reportdata = {
                    "userid": userid,
                    "time": time,
                    "location": location,
                    "filename": filename,
                    "file_type": file_type,
                    "type": file_type,
                    "file_id": file_id,
                    "file_url": file_url
                }

                report_data.insert_one(reportdata)

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