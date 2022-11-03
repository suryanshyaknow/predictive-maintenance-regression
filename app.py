from flask import Flask, render_template, request, jsonify
from flask_cors import CORS, cross_origin
from regress import LModel
import logging as lg
from logger import Log

app = Flask(__name__)  # intializing the Flask app

# Executing the logger class
Log()

@app.route('/', methods=['GET'])
@cross_origin()
def homepage():
    try:
        return render_template("index.html")
    except Exception as e:
        lg.exception(e)
        error_m1 = "There's some error loading the Analytics Report."
        error_m2 = "Kindly try again after some time!"
        return render_template('errors.html', error_m1=error_m1, error_m2=error_m2)

@app.route('/dataset+analytics', methods=['Get'])
@cross_origin()
def dataset_analytics():
    try:
        lg.info("rendering dataset analytics..")
        return render_template("dataset_report.html")
    except Exception as e:
        lg.exception(e)

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
            prediction = lm.predict(process_temp, rot_speed, torque, tool_wear, machine_f, twf, hdf, pwf, osf, rnf)

        except Exception as e:
            lg.exception(e)
            error_m1 = "It's either that you have rendered some inputs empty or its values are out of the range."
            error_m2 = "Please try again with apt values!"
            return render_template('errors.html', error_m1=error_m1, error_m2=error_m2)

        else:
            lg.info("rendering results..")
            return render_template('results.html', prediction=round(prediction, 3), accuracy=lm.accuracy())


if __name__ == '__main__':
    app.run(debug=True)
