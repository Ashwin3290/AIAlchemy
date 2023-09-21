from flask import Flask, request, jsonify
from flask_restful import Api,Resource,reqparse


app = Flask((__name__))
api=Api(app)
student_roll =reqparse.RequestParser()
student_roll.add_argument("roll",type=int,help="none")

class roll(Resource):
    def post(self):
        args=student_roll.parse_args()
        return {"roll":args["roll"]}

api.add_resource(roll,"/roll")
student_name=reqparse.RequestParser()
student_name.add_argument("name",type=str,help="Send student name")

class name(Resource):
    def post(self):
        args=student_name.parse_args()
        processed=args["name"].split("-")
        return jsonify(processed)

api.add_resource(name,"/name")
if __name__ == '__main__':
    app.run(debug=True)