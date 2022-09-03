# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
# .ml/scripts/activate

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS, cross_origin
from regress import LModel
import logging as lg
from logger import Log

app = Flask(__name__)  # intializing our Flask app

# executing the logger class
Log()

@app.route('/', methods=['GET'])
@cross_origin()
def homepage(): 
    return render_template("index.html")


@app.route('/dataset+report', methods=['Get'])
@cross_origin()
def dataset_report():
    return render_template("dataset_report.html")


@app.route('/prediction+results', methods=['POST', 'GET'])
@cross_origin()
def main():
    if request.method == 'POST':
        try:
            # fetching the predictors
            lg.info("fetching the predictors..")
            process_temp = float(request.form['process_temp'])
            rot_speed = int(request.form['rot_speed'])
            torque = float(request.form['torque'])
            tool_wear = float(request.form['tool_wear'])
            machine_f = int(request.form['machine_f'])
            twf = int(request.form['twf'])
            hdf = int(request.form['hdf'])
            pwf = int(request.form['pwf'])
            osf = int(request.form['osf'])
            rnf = int(request.form['rnf'])
            lg.info("predictors values fetched!")

            # loading our Regression model
            lm = LModel()
            lm.build()
            print("done !")
            prediction = lm.predict(process_temp, rot_speed, torque, tool_wear, machine_f, twf, hdf, pwf, osf, rnf)

        except Exception as e:
            return render_template('errors.html', errors=e)

        else:
            return render_template('results.html', prediction=round(prediction, 3), accuracy=lm.accuracy())


if __name__ == '__main__':
    app.run(debug=True)
    # app.run(host='127.0.0.1', port=8001, debug=True)
