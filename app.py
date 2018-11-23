from flask import Flask, jsonify, request
# from Model.Dataseed import tasks
# from Model.Petal import Petal
from flasgger import Swagger
from sklearn.externals import joblib
import numpy as np
from flask_cors import CORS


app = Flask(__name__)
Swagger(app)
CORS(app)

# @app.route('/get/task', methods=['GET'])
# def getTask():
#     return jsonify({'task': tasks})


@app.route('/input/task', methods=['POST'])
def predict():
    """
    Ini adalah endpoint untuk memperbaiki IRIS
    ---
    tags:
        - Rest Controller
    parameters:
      - name: body
        in: body
        required: true
        Schema:
          id: Petal
          required:
            - petalLength
            - petalWidth
            - sepalLength
            - sepalWidth
          properties:
            petalLength:
              type: int
              description: Please input with valid sepal and petal length-width.
              default: 0
            petalWidth:
              type: int
              description: Please input with valid sepal and petal length-width.
              default: 0
            sepalLength:
              type: int
              description: Please input with valid sepal and petal length-width.
              default: 0
            sepalWidth:
              type: int
              description: Please input with valid sepal and petal length-width.
              default: 0
    responses:
        200:
            description: Success Input
    """
    new_task = request.get_json()

    petalLength = new_task['petalLength']
    petalWidth = new_task['petalWidth']
    sepalLength = new_task['sepalLength']
    sepalWidth = new_task['sepalWidth']

    x_new = np.array([[petalLength,petalWidth,sepalLength,sepalWidth]])
    clf = joblib.load('knnClassifier.pkl')
    resultPredict = clf[0].predict(x_new)

    return jsonify({'message':format(clf[1].target_names[resultPredict])})


# @app.route('/update/task/<int:id>', methods=['PUT'])
# def updateTask(id):
#     new_task = request.get_json()
#
#     petalLength = new_task['petalLength']
#     petalWidth = new_task['petalWidth']
#     sepalLength = new_task['sepalLength']
#     sepalWidth = new_task['sepalWidth']
#
#     newPetal = Petal(petalLength, petalWidth, sepalLength, sepalWidth)
#
#     tasks[id] = newPetal.__dict__
#
#     return jsonify({'message': 'success update'})
#
#
# @app.route('/delete/task/<int:id>', methods=['DELETE'])
# def deleteTask(id):
#     del tasks[id]
#
#     return jsonify({'message': 'success delete'})
#
