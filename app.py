from flask import Flask,request
from flask_restful import Resource, Api
import pickle
import pandas as pd
from flask_cors import CORS


app = Flask(__name__)
#
CORS(app)
# creating an API object
api = Api(app)

#prediction api call
class prediction(Resource):
    def get(self, budget):
        print(budget)
        budget = [int(budget)]
        
        # ✅ Match training feature name
        df = pd.DataFrame(budget, columns=['Marketing Budget (X) in Thousands'])
        
        model = pickle.load(open('simple_linear_regression.pkl', 'rb'))
        prediction = model.predict(df)
        prediction = int(prediction[0])
        return str(prediction)



api.add_resource(prediction, '/prediction/<int:budget>')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
