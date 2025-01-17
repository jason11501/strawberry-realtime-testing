# # """
# # Script to run inference on videos using ONNX model.
# # `--input` takes the path to a video.

# # USAGE:
# # python onnx_inference_video.py --input ../inference_data/video_4_trimmed_1.mp4 --weights weights/fasterrcnn_resnet18.onnx --data data_configs/voc.yaml --show --imgsz 640
# # """

# import numpy as np
# import cv2
# import torch
# import glob as glob
# import os
# import time
# import argparse
# import yaml
# import onnxruntime

# from utils.general import set_infer_dir
# from utils.annotations import (
#     inference_annotations, 
#     annotate_fps, 
#     convert_detections,
#     convert_pre_track,
#     convert_post_track
# )
# from utils.transforms import infer_transforms, resize
# from deep_sort_realtime.deepsort_tracker import DeepSort
# from utils.logging import LogJSON
# from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
# from websockets.exceptions import ConnectionClosed
# from fastapi.templating import Jinja2Templates
# import uvicorn    #  WEBSOCKET
# import asyncio
# app = FastAPI()
# def read_return_video_data(video_path):
#     cap = cv2.VideoCapture(0)
#     # Get the video's frame width and height
#     frame_width = int(cap.get(3))
#     frame_height = int(cap.get(4))
#     assert (frame_width != 0 and frame_height !=0), 'Please check video path...'
#     return cap, frame_width, frame_height

# def to_numpy(tensor):
#         return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# def parse_opt():
#         # Construct the argument parser.
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         '-i', '--input', 
#         help='path to input video',
#     )
#     parser.add_argument(
#         '--data', 
#         default=None,
#         help='(optional) path to the data config file'
#     )
#     parser.add_argument(
#         '-m', '--model', 
#         default=None,
#         help='name of the model'
#     )
#     parser.add_argument(
#         '-w', '--weights', 
#         default=None,
#         help='path to trained checkpoint weights if providing custom YAML file'
#     )
#     parser.add_argument(
#         '-th', '--threshold', 
#         default=0.3, 
#         type=float,
#         help='detection threshold'
#     )
#     parser.add_argument(
#         '-si', '--show',  
#         action='store_true',
#         help='visualize output only if this argument is passed'
#     )
#     parser.add_argument(
#         '-mpl', '--mpl-show', 
#         dest='mpl_show', 
#         action='store_true',
#         help='visualize using matplotlib, helpful in notebooks'
#     )
#     parser.add_argument(
#         '-ims', '--imgsz', 
#         default=None,
#         type=int,
#         help='resize image to, by default use the original frame/image size'
#     )
#     parser.add_argument(
#         '-nlb', '--no-labels',
#         dest='no_labels',
#         action='store_true',
#         help='do not show labels during on top of bounding boxes'
#     )
#     parser.add_argument(
#         '--classes',
#         nargs='+',
#         type=int,
#         default=None,
#         help='filter classes by visualization, --classes 1 2 3'
#     )
#     parser.add_argument(
#         '--track',
#         action='store_true'
#     )
#     parser.add_argument(
#         '--log-json',
#         dest='log_json',
#         action='store_true',
#         help='store a json log file in COCO format in the output directory'
#     )
#     args = vars(parser.parse_args())
#     return args

# def main(args):
#     np.random.seed(42)
#     if args['track']: # Initialize Deep SORT tracker if tracker is selected.
#         tracker = DeepSort(max_age=30)
#     # Load model.
#     ort_session = onnxruntime.InferenceSession(
#         "/Users/c/Downloads/fasterrcnn.onnx", providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
#     )
    
#     NUM_CLASSES = 2
#     CLASSES = CLASSES= [
#     '__background__',
#     'strawberry'
# ]

#     COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

#     # Define the detection threshold any detection having
#     # score below this will be discarded.
#     detection_threshold = args['threshold']

#     cap, frame_width, frame_height = read_return_video_data(0)
#     # templates = Jinja2Templates(directory="templates")

#     # @app.get('/')
#     # def index(request: Request):
#     #     return templates.TemplateResponse("index.html", {"request": request})

#     if args['imgsz'] != None:
#         RESIZE_TO = args['imgsz']
#     else:
#         RESIZE_TO = frame_width

#     if args['log_json']:
#         log_json = LogJSON(os.path.join(OUT_DIR, 'log.json'))

#     frame_count = 0 # To count total frames.
#     total_fps = 0 # To get the final frames per second.
#     from fastapi.responses import HTMLResponse
#     @app.get("/", response_class=HTMLResponse)
#     async def home():
#         """
#         Serve a webpage with JavaScript to access webcam and send frames via WebSocket.
#         """
#         return """
#         <!DOCTYPE html>
#         <html>
#         <head>
#             <title>Webcam Stream</title>
#         </head>
#         <body>
#             <h1>Webcam Stream</h1>
#             <video id="video" autoplay playsinline></video>
#             <canvas id="canvas" style="display: none;"></canvas>
#             <script>
#                 const video = document.getElementById('video');
#                 const canvas = document.getElementById('canvas');
#                 const ctx = canvas.getContext('2d');

#                 // Access webcam
#                 navigator.mediaDevices.getUserMedia({ video: true })
#                     .then(stream => {
#                         video.srcObject = stream;
#                     })
#                     .catch(err => {
#                         console.error("Error accessing webcam:", err);
#                     });

#                 // WebSocket connection
#                 const ws = new WebSocket("ws://<PUBLIC_URL>/ws".replace("<PUBLIC_URL>", window.location.hostname + ":8000"));

#                 ws.onopen = () => {
#                     console.log("WebSocket connection established");
#                     setInterval(() => {
#                         canvas.width = video.videoWidth;
#                         canvas.height = video.videoHeight;
#                         ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

#                         // Send the frame as a Blob
#                         canvas.toBlob(blob => {
#                             if (blob) {
#                                 ws.send(blob);
#                             }
#                         }, "image/jpeg");
#                     }, 100); // Send every 100ms
#                 };

#                 ws.onmessage = event => {
#                     console.log("Server response:", event.data);
#                 };

#                 ws.onclose = () => {
#                     console.log("WebSocket connection closed");
#                 };
#             </script>
#         </body>
#         </html>
#         """
#     @app.websocket("/ws")
#     async def get_stream(websocket: WebSocket):
#         await websocket.accept()
#         try:
#     # read until end of video
#             while(cap.isOpened()):
#                 # capture each frame of the video
#                 ret, frame = cap.read()
#                 if not ret:
#                     break
#                 else:
#                     orig_frame = frame.copy()
#                     frame = resize(frame, RESIZE_TO, square=True)
#                     image = frame.copy()
#                     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#                     image = infer_transforms(image)
#                     # Add batch dimension.
#                     image = torch.unsqueeze(image, 0)
#                     # Get the start time.
#                     start_time = time.time()
#                     preds = ort_session.run(
#                         None, {ort_session.get_inputs()[0].name: to_numpy(image)}
#                     )
#                     forward_end_time = time.time()

#                     forward_pass_time = forward_end_time - start_time
#                     # Get the current fps.
#                     fps = 1 / (forward_pass_time)
#                     # Add `fps` to `total_fps`.
#                     total_fps += fps
#                     # Increment frame count.
#                     frame_count += 1
#                     outputs = {}
#                     outputs['boxes'] = torch.tensor(preds[0])
#                     outputs['labels'] = torch.tensor(preds[1])
#                     outputs['scores'] = torch.tensor(preds[2])
#                     outputs = [outputs]

#                     # Log to JSON?
#                     if args['log_json']:
#                         log_json.update(frame, save_name, outputs[0], CLASSES)

#                     # Carry further only if there are detected boxes.
#                     if len(outputs[0]['boxes']) != 0:
#                         draw_boxes, pred_classes, scores,_ = convert_detections(
#                             outputs, detection_threshold, CLASSES, args
#                         )
#                         if args['track']:
#                             tracker_inputs = convert_pre_track(
#                                 draw_boxes, pred_classes, scores
#                             )
#                             # Update tracker with detections.
#                             tracks = tracker.update_tracks(tracker_inputs, frame=frame)
#                             draw_boxes, pred_classes, scores = convert_post_track(tracks) 
#                         frame = inference_annotations(
#                             draw_boxes, 
#                             pred_classes, 
#                             scores,
#                             CLASSES, 
#                             COLORS, 
#                             orig_frame, 
#                             frame,
#                             args
#                         )
#                     else:
#                         frame = orig_frame
#                     frame = annotate_fps(frame, fps)

#                     final_end_time = time.time()
#                     forward_and_annot_time = final_end_time - start_time
#                     print_string = f"Frame: {frame_count}, Forward pass FPS: {fps:.3f}, "
#                     print_string += f"Forward pass time: {forward_pass_time:.3f} seconds, "
#                     print_string += f"Forward pass + annotation time: {forward_and_annot_time:.3f} seconds"
#                     print(print_string)            
#                     # out.write(frame)
#                     # if args['show']:
#                     cv2.imshow('Prediction', frame)
#                     # frame = frame[0].plot()
#                     # cv2.rectangle(frame,(10,5),(40,300),(255,0,0),2)
#                     _, buffer = cv2.imencode('.jpg', frame)
#                     # await websocket.send_text("WEB CAM PHAM XUAN KY")
#                     await websocket.send_bytes(buffer.tobytes())
#                     # Press `q` to exit
#                     if cv2.waitKey(1) & 0xFF == ord('q'):
#                         break
#                 await asyncio.sleep(0.03)   
                
#         except (WebSocketDisconnect, ConnectionClosed):
#             print("WebSocket/Client disconnected")
#     # Release VideoCapture().
#     cap.release()
#     # Close all frames and video windows.
#     cv2.destroyAllWindows()
#     from pyngrok import ngrok
#     ngrok.set_auth_token("2qNO4pkphxEwuI6fyH3SWEcEwNC_iuzBKEyGc5DgtAnDo4KV")  # Replace with your actual ngrok token
#     public_url = ngrok.connect(8000)
#     uvicorn.run(app, port=8000)
#     # Save JSON log file.
#     if args['log_json']:
#         log_json.save(os.path.join(OUT_DIR, 'log.json'))

#     # Calculate and print the average FPS.
#     # avg_fps = total_fps / frame_count
#     # print(f"Average FPS: {avg_fps:.3f}")

# if __name__ == '__main__':
#     args = parse_opt()
#     main(args)
"""
Script to run inference on videos using ONNX model.
`--input` takes the path to a video.

USAGE:
python onnx_inference_video.py --input ../inference_data/video_4_trimmed_1.mp4 --weights weights/fasterrcnn_resnet18.onnx --data data_configs/voc.yaml --show --imgsz 640
"""

import numpy as np
import cv2
import torch
import glob as glob
import os
import time
import argparse
import yaml
import onnxruntime
from flask import Flask, render_template, request, redirect, send_file, url_for, Response
from utils.general import set_infer_dir
from utils.annotations import (
    inference_annotations, 
    annotate_fps, 
    convert_detections,
    convert_pre_track,
    convert_post_track
)
from utils.transforms import infer_transforms, resize
from deep_sort_realtime.deepsort_tracker import DeepSort
from utils.logging import LogJSON
from werkzeug.utils import secure_filename, send_from_directory
app = Flask(__name__)
@app.route("/")
def hello_world():
    return render_template('index.html')
# function for accessing rtsp stream
@app.route("/rtsp_feed")
def rtsp_feed():
    cap = cv2.VideoCapture('rtsp://admin:hello123@192.168.29.126:554/cam/realmonitor?channel=1&subtype=0')
    return render_template('index.html')

# Function to start webcam and detect objects
@app.route("/webcam_feed")
def webcam_feed():
    #source = 0
    cap = cv2.VideoCapture(0)
    return render_template('index.html')
# function to display the detected objects video on html page
@app.route("/video_feed")
def video_feed():
    args = parse_opt()
    return Response(get_frame(args),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
#The display function is used to serve the image or video from the folder_path directory.
@app.route('/<path:filename>')
def display(filename):
    folder_path = 'outputs/inference/'
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]    
    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))    
    directory = folder_path+'/'+latest_subfolder
    print("printing directory: ",directory)  
    filename = predict_img.imgpath
    file_extension = filename.rsplit('.', 1)[1].lower()
    #print("printing file extension from display function : ",file_extension)
    environ = request.environ
    if file_extension == 'jpg':      
        return send_from_directory(directory,filename,environ)

    elif file_extension == 'mp4':
        return render_template('index.html')

    else:
        return "Invalid file format"

@app.route("/", methods=["GET", "POST"])
def predict_img():
    if request.method == "POST":
        if 'file' in request.files:
            f = request.files['file']
            basepath = os.path.dirname(__file__)
            filepath = os.path.join(basepath,'uploads',f.filename)
            print("upload folder is ", filepath)
            f.save(filepath)
            
            predict_img.imgpath = f.filename
            print("printing predict_img :::::: ", predict_img)

            file_extension = f.filename.rsplit('.', 1)[1].lower()    
            if file_extension == 'jpg':
                process = Popen(["python", "detect.py", '--source', filepath, "--weights","best_246.pt"], shell=True)
                process.wait()
                
                
            elif file_extension == 'mp4':
                process = Popen(["python", "detect.py", '--source', filepath, "--weights","best_246.pt"], shell=True)
                process.communicate()
                process.wait()

            
    folder_path = 'outputs/inference/'
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]    
    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))    
    image_path = folder_path+'/'+latest_subfolder+'/'+f.filename 
    return render_template('index.html', image_path=image_path)
    #return "done"


def read_return_video_data(video_path):
    cap = cv2.VideoCapture(0)
    # Get the video's frame width and height
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    assert (frame_width != 0 and frame_height !=0), 'Please check video path...'
    return cap, frame_width, frame_height

def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def get_frame(args):
    
    np.random.seed(42)
    # if args['track']: # Initialize Deep SORT tracker if tracker is selected.
    #     tracker = DeepSort(max_age=30)
    # Load model.
    ort_session = onnxruntime.InferenceSession(
        "/Users/c/Downloads/fasterrcnn.onnx", providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    
    NUM_CLASSES = 2
    CLASSES = CLASSES= [
    '__background__',
    'strawberry'
]

    # OUT_DIR = set_infer_dir()
    # VIDEO_PATH = None
    # if args['input'] == None:
    #     VIDEO_PATH = data_configs['video_path']
    # else:
    #     VIDEO_PATH = args['input']
    # assert VIDEO_PATH is not None, 'Please provide path to an input video...'
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    # Define the detection threshold any detection having
    # score below this will be discarded.
    detection_threshold = 0.3

    cap, frame_width, frame_height = read_return_video_data(0)

    # save_name = VIDEO_PATH.split(os.path.sep)[-1].split('.')[0]
    # Define codec and create VideoWriter object.
    # out = cv2.VideoWriter(f"{OUT_DIR}/{save_name}.mp4", 
                        # cv2.VideoWriter_fourcc(*'mp4v'), 30, 
                        # (frame_width, frame_height))
    # if args['imgsz'] != None:
    #     RESIZE_TO = args['imgsz']
    # else:
    RESIZE_TO = frame_width

    # if args['log_json']:
    #     log_json = LogJSON(os.path.join(OUT_DIR, 'log.json'))

    frame_count = 0 # To count total frames.
    total_fps = 0 # To get the final frames per second.

    # read until end of video
    while(cap.isOpened()):
        # capture each frame of the video
        ret, frame = cap.read()
        if ret:
            orig_frame = frame.copy()
            frame = resize(frame, RESIZE_TO, square=True)
            image = frame.copy()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = infer_transforms(image)
            # Add batch dimension.
            image = torch.unsqueeze(image, 0)
            # Get the start time.
            start_time = time.time()
            preds = ort_session.run(
                None, {ort_session.get_inputs()[0].name: to_numpy(image)}
            )
            forward_end_time = time.time()

            forward_pass_time = forward_end_time - start_time
            # Get the current fps.
            fps = 1 / (forward_pass_time)
            # Add `fps` to `total_fps`.
            total_fps += fps
            # Increment frame count.
            frame_count += 1
            outputs = {}
            outputs['boxes'] = torch.tensor(preds[0])
            outputs['labels'] = torch.tensor(preds[1])
            outputs['scores'] = torch.tensor(preds[2])
            outputs = [outputs]

            
            # Carry further only if there are detected boxes.
            if len(outputs[0]['boxes']) != 0:
                draw_boxes, pred_classes, scores,_ = convert_detections(
                    outputs, detection_threshold, CLASSES,args
                )
                frame = inference_annotations(
                    draw_boxes, 
                    pred_classes, 
                    scores,
                    CLASSES, 
                    COLORS, 
                    orig_frame, 
                    frame,
                    args
                )
            else:
                frame = orig_frame
            frame = annotate_fps(frame, fps)

            final_end_time = time.time()
            forward_and_annot_time = final_end_time - start_time
            print_string = f"Frame: {frame_count}, Forward pass FPS: {fps:.3f}, "
            print_string += f"Forward pass time: {forward_pass_time:.3f} seconds, "
            print_string += f"Forward pass + annotation time: {forward_and_annot_time:.3f} seconds"
            print(print_string)            
            # out.write(frame)
            # if args['show']:
            # cv2.imshow('Prediction', frame)
            ret, jpeg = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')   
            time.sleep(0.1)  #control the frame rate to display one frame every 100 milliseconds: 

            # Press `q` to exit
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

        else:
            break

    # Release VideoCapture().
    cap.release()
    # Close all frames and video windows.
    cv2.destroyAllWindows()

    # Save JSON log file.
    # if args['log_json']:
    #     log_json.save(os.path.join(OUT_DIR, 'log.json'))

    # Calculate and print the average FPS.
    # avg_fps = total_fps / frame_count
    # print(f"Average FPS: {avg_fps:.3f}")
def parse_opt():
        # Construct the argument parser.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input', 
        help='path to input video',
    )
    parser.add_argument(
        '--data', 
        default=None,
        help='(optional) path to the data config file'
    )
    parser.add_argument(
        '-m', '--model', 
        default=None,
        help='name of the model'
    )
    parser.add_argument(
        '-w', '--weights', 
        default=None,
        help='path to trained checkpoint weights if providing custom YAML file'
    )
    parser.add_argument(
        '-th', '--threshold', 
        default=0.3, 
        type=float,
        help='detection threshold'
    )
    parser.add_argument(
        '-si', '--show',  
        action='store_true',
        help='visualize output only if this argument is passed'
    )
    parser.add_argument(
        '-mpl', '--mpl-show', 
        dest='mpl_show', 
        action='store_true',
        help='visualize using matplotlib, helpful in notebooks'
    )
    parser.add_argument(
        '-ims', '--imgsz', 
        default=None,
        type=int,
        help='resize image to, by default use the original frame/image size'
    )
    parser.add_argument(
        '-nlb', '--no-labels',
        dest='no_labels',
        action='store_true',
        help='do not show labels during on top of bounding boxes'
    )
    parser.add_argument(
        '--classes',
        nargs='+',
        type=int,
        default=None,
        help='filter classes by visualization, --classes 1 2 3'
    )
    parser.add_argument(
        '--track',
        action='store_true'
    )
    parser.add_argument(
        '--log-json',
        dest='log_json',
        action='store_true',
        help='store a json log file in COCO format in the output directory'
    )
    args = vars(parser.parse_args())
    return args

if __name__ == '__main__':
    args = parse_opt()
    get_frame(args)
    app.run(host='0.0.0.0', port='5000', debug=True)