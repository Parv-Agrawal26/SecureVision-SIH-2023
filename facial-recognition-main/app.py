# app.py

from flask import Flask, render_template
import subprocess
import cv2
import numpy as np
import threading

app = Flask(__name__)


@app.route('/crowd_management')
def index():
    return render_template('index.html', css_file='stylescrowd.css')


@app.route('/run_crowd_management')
def run_crowd_management():
    try:
        # Execute the Python script in the background
        subprocess.Popen(['python', 'crowd_management_script.py'])
        return "Crowd management script is running in the background."
    except Exception as e:
        return f"Error: {str(e)}"


@app.route('/object_detection')
def index2():
    return render_template('index2.html', css_file='stylesobject.css')


@app.route('/run_object_detection')
def run_object_detection():
    try:
        # Execute the Python script in the background
        subprocess.Popen(['python', 'object_detection_script.py'])
        return "Object Detection script is running in the background."
    except Exception as e:
        return f"Error: {str(e)}"


@app.route('/facial_req')
def index3():
    return render_template('index3.html', css_file='stylescriminal.css')


@app.route('/run_facial_req')
def run_facial_req():
    try:
        # Execute the Python script in the background
        subprocess.Popen(['python', 'facial_req_script.py'])
        return "Facial Recognition script is running in the background."
    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == '__main__':
    app.run(debug=True)
