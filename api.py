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
=========
student_name=reqparse.RequestParser()
student_name.add_argument("name",type=str,help="Send student name")
>>>>>>>>> Temporary merge branch 2

class name(Resource):
    def post(self):
        args=student_name.parse_args()
        processed=args["name"].split("-")
        return jsonify(processed)

api.add_resource(name,"/name")
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class FileUpload(Resource):
    def post(self):
        try:
            uploaded_file = request.files['file']  
            if uploaded_file and allowed_file(uploaded_file.filename):
                if len(uploaded_file.read()) > MAX_CONTENT_LENGTH:
                    return jsonify({
                        "status": "error",
                        "message": "File size exceeds the maximum allowed size (30MB)"
                    })

                uploaded_file.seek(0)
                file_id = fs.put(uploaded_file.read(), filename=secure_filename(uploaded_file.filename),
                                 content_type=uploaded_file.content_type)

                return jsonify({
                    "status": "success",
                    "message": "File uploaded and saved to MongoDB",
                    "file_id": str(file_id)
                })
            else:
                return jsonify({
                    "status": "error",
                    "message": "Invalid file format. Allowed formats: XLSX, CSV"
                })
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": str(e)
            })


if __name__ == '__main__':
    app.run(debug=True)