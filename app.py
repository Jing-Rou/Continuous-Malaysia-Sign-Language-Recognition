from flask import *
from transformers import T5Tokenizer, TFT5ForConditionalGeneration
from threading import Thread
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import mediapipe as mp
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import datetime
import time
import string
import re
import os

global rec_frame, rec, out, file_date, file_path, camera, tut_video_path, content
file_path = ''
file_date = ''
rec = 0
camera = 0
ROWS_PER_FRAME = 543  # number of landmarks per frame
tut_video_path = ''
content = ''

app = Flask(__name__, template_folder='pages', static_folder='static')

# opencv
cap = cv2.VideoCapture(0)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5, min_tracking_confidence=0.5)

# deployment
# read trained model
interpreter = tf.lite.Interpreter("model/model_3.tflite")
found_signatures = list(interpreter.get_signature_list().keys())
prediction_fn = interpreter.get_signature_runner("serving_default")

train = pd.read_csv('D:/NEW/dataset/train_new_1.csv')
# Add ordinally Encoded Sign (assign number to each sign name)
train['sign_ord'] = train['sign'].astype('category').cat.codes
# Dictionaries to translate sign <-> ordinal encoded sign
SIGN2ORD = train[['sign', 'sign_ord']].set_index('sign').squeeze().to_dict()
ORD2SIGN = train[['sign_ord', 'sign']].set_index(
    'sign_ord').squeeze().to_dict()

# text preprocessing model
tokenizer = T5Tokenizer.from_pretrained('SJ-Ray/Re-Punctuate')
model = TFT5ForConditionalGeneration.from_pretrained('SJ-Ray/Re-Punctuate')

# animation initialization
plt.rcParams['animation.ffmpeg_path'] = 'C:/Users/User/ffmpeg/bin/ffmpeg.exe'

def record(out):
    global rec_frame
    while (rec):
        time.sleep(0.05)
        out.write(rec_frame)

def create_frame_landmark_df(results, frameNo):
    """
        Takes the results from mediapipe and create a dataframe of the landmarks
        Inputs:
            results: mediapipe results object
            frame: number of frame
    """

    # must ensure the total number of row per frame is 543
    # Define the total number of landmark indices for each type
    total_face_landmarks = 468
    total_pose_landmarks = 33
    total_left_hand_landmarks = 21
    total_right_hand_landmarks = 21

    # Create DataFrames for each type
    face_landmark = pd.DataFrame({
        'landmark_index': range(total_face_landmarks),
        'type': 'face'
    })

    left_hand_landmark = pd.DataFrame({
        'landmark_index': range(total_left_hand_landmarks),
        'type': 'left_hand'
    })

    pose_landmark = pd.DataFrame({
        'landmark_index': range(total_pose_landmarks),
        'type': 'pose'
    })

    right_hand_landmark = pd.DataFrame({
        'landmark_index': range(total_right_hand_landmarks),
        'type': 'right_hand'
    })

    if results.face_landmarks is not None:
        for i, point in enumerate(results.face_landmarks.landmark):
            face_landmark.loc[i, ['x', 'y', 'z']] = [point.x, point.y, point.z]

    if results.pose_landmarks is not None:
        for i, point in enumerate(results.pose_landmarks.landmark):
            pose_landmark.loc[i, ['x', 'y', 'z']] = [point.x, point.y, point.z]

    if results.left_hand_landmarks is not None:
        for i, point in enumerate(results.left_hand_landmarks.landmark):
            left_hand_landmark.loc[i, ['x', 'y', 'z']] = [
                point.x, point.y, point.z]

    if results.right_hand_landmarks is not None:
        for i, point in enumerate(results.right_hand_landmarks.landmark):
            right_hand_landmark.loc[i, ['x', 'y', 'z']] = [
                point.x, point.y, point.z]

    landmarks = pd.concat([face_landmark, left_hand_landmark,
                          pose_landmark, right_hand_landmark]).set_index('landmark_index')
    landmarks = landmarks.assign(frame=int(frameNo))

    return landmarks

def do_capture_loop(v_path, target_resolution=(160, 120)):
    all_landmarks_list = []
    all_landmarks = pd.DataFrame()

    # For webcam input:
    cap = cv2.VideoCapture(v_path)
    frameNo = 0

    while cap.isOpened():
        frameNo += 1

        success, image = cap.read()

        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            break

        # Resize the frame to the target resolution
        image = cv2.resize(image, target_resolution)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)

        # create landmarks
        landmarks = create_frame_landmark_df(results, frameNo)
        # Append 'landmarks' to 'all_landmarks_list'
        all_landmarks_list.append(landmarks)

    all_landmarks = pd.concat(all_landmarks_list, ignore_index=True)

    return all_landmarks

def generate_frames():  # generator function for frame by frame camera capture
    global cap, rec_frame, file_path

    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            pass

        # recording
        if (rec):

            # frame = cv2.resize(frame, (320, 240))
            rec_frame = frame

            frame = cv2.putText(
                frame, "Recording...", (370, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)

            frame = cv2.resize(frame, (640, 480))

        # draw landmark only
        if (camera):
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            # real-time detection
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame)

            # Draw landmark annotation on the image.
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            frame = frame * 0

            # Left Hand
            mp_drawing.draw_landmarks(frame,                                                  # image to draw
                                      results.left_hand_landmarks,                              # model output
                                      mp_holistic.HAND_CONNECTIONS,                             # hand connections
                                      mp_drawing_styles.get_default_hand_landmarks_style(),
                                      mp_drawing_styles.get_default_hand_connections_style())
            # Right Hand
            mp_drawing.draw_landmarks(frame,
                                      results.right_hand_landmarks,
                                      mp_holistic.HAND_CONNECTIONS,
                                      mp_drawing_styles.get_default_hand_landmarks_style(),
                                      mp_drawing_styles.get_default_hand_connections_style())

            # Face
            mp_drawing.draw_landmarks(frame,
                                      results.face_landmarks,
                                      mp_holistic.FACEMESH_CONTOURS,
                                      landmark_drawing_spec=None,
                                      connection_drawing_spec=mp_drawing_styles
                                      .get_default_face_mesh_contours_style())

            # Pose
            mp_drawing.draw_landmarks(frame,
                                      results.pose_landmarks,
                                      mp_holistic.POSE_CONNECTIONS,
                                      landmark_drawing_spec=mp_drawing_styles
                                      .get_default_pose_landmarks_style())
            time.sleep(0.1 / fps)

        try:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            pass

def get_prediction(pq_path, prediction_fn):
    data_columns = ['x', 'y', 'z']
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))

    xyz_np = data.astype(np.float32)

    prediction = prediction_fn(inputs=xyz_np)
    top_n_indices = np.argpartition(-prediction['outputs'], 3)[:3]

    top_n_predictions = []
    for index in top_n_indices:
        sign = ORD2SIGN[index]
        pred_conf = prediction['outputs'][index]
        top_n_predictions.append((sign, index, pred_conf))

    top_n_predictions.sort(key=lambda x: x[2], reverse=True)

    # Extract only the signs from top_n_predictions
    top_n_signs = [item[0] for item in top_n_predictions]

    for i, (sign, index, pred_conf) in enumerate(top_n_predictions):
        print(
            f'Top {i + 1} Prediction Sign: {sign} [{index}], CONFIDENCE {pred_conf:0.4}')

    return top_n_signs

def get_hand_points(hand):
    x = [[hand.iloc[0].x, hand.iloc[1].x, hand.iloc[2].x, hand.iloc[3].x, hand.iloc[4].x], # Thumb
         [hand.iloc[5].x, hand.iloc[6].x, hand.iloc[7].x, hand.iloc[8].x], # Index
         [hand.iloc[9].x, hand.iloc[10].x, hand.iloc[11].x, hand.iloc[12].x], 
         [hand.iloc[13].x, hand.iloc[14].x, hand.iloc[15].x, hand.iloc[16].x], 
         [hand.iloc[17].x, hand.iloc[18].x, hand.iloc[19].x, hand.iloc[20].x], 
         [hand.iloc[0].x, hand.iloc[5].x, hand.iloc[9].x, hand.iloc[13].x, hand.iloc[17].x, hand.iloc[0].x]]

    y = [[hand.iloc[0].y, hand.iloc[1].y, hand.iloc[2].y, hand.iloc[3].y, hand.iloc[4].y],  #Thumb
         [hand.iloc[5].y, hand.iloc[6].y, hand.iloc[7].y, hand.iloc[8].y], # Index
         [hand.iloc[9].y, hand.iloc[10].y, hand.iloc[11].y, hand.iloc[12].y], 
         [hand.iloc[13].y, hand.iloc[14].y, hand.iloc[15].y, hand.iloc[16].y], 
         [hand.iloc[17].y, hand.iloc[18].y, hand.iloc[19].y, hand.iloc[20].y], 
         [hand.iloc[0].y, hand.iloc[5].y, hand.iloc[9].y, hand.iloc[13].y, hand.iloc[17].y, hand.iloc[0].y]] 
    return x, y

def get_pose_points(pose):
    x = [[pose.iloc[8].x, pose.iloc[6].x, pose.iloc[5].x, pose.iloc[4].x, pose.iloc[0].x, pose.iloc[1].x, pose.iloc[2].x, pose.iloc[3].x, pose.iloc[7].x], 
         [pose.iloc[10].x, pose.iloc[9].x], 
         [pose.iloc[22].x, pose.iloc[16].x, pose.iloc[20].x, pose.iloc[18].x, pose.iloc[16].x, pose.iloc[14].x, pose.iloc[12].x, 
          pose.iloc[11].x, pose.iloc[13].x, pose.iloc[15].x, pose.iloc[17].x, pose.iloc[19].x, pose.iloc[15].x, pose.iloc[21].x], 
         [pose.iloc[12].x, pose.iloc[24].x, pose.iloc[26].x, pose.iloc[28].x, pose.iloc[30].x, pose.iloc[32].x, pose.iloc[28].x], 
         [pose.iloc[11].x, pose.iloc[23].x, pose.iloc[25].x, pose.iloc[27].x, pose.iloc[29].x, pose.iloc[31].x, pose.iloc[27].x], 
         [pose.iloc[24].x, pose.iloc[23].x]
        ]

    y = [[pose.iloc[8].y, pose.iloc[6].y, pose.iloc[5].y, pose.iloc[4].y, pose.iloc[0].y, pose.iloc[1].y, pose.iloc[2].y, pose.iloc[3].y, pose.iloc[7].y], 
         [pose.iloc[10].y, pose.iloc[9].y], 
         [pose.iloc[22].y, pose.iloc[16].y, pose.iloc[20].y, pose.iloc[18].y, pose.iloc[16].y, pose.iloc[14].y, pose.iloc[12].y, 
          pose.iloc[11].y, pose.iloc[13].y, pose.iloc[15].y, pose.iloc[17].y, pose.iloc[19].y, pose.iloc[15].y, pose.iloc[21].y], 
         [pose.iloc[12].y, pose.iloc[24].y, pose.iloc[26].y, pose.iloc[28].y, pose.iloc[30].y, pose.iloc[32].y, pose.iloc[28].y], 
         [pose.iloc[11].y, pose.iloc[23].y, pose.iloc[25].y, pose.iloc[27].y, pose.iloc[29].y, pose.iloc[31].y, pose.iloc[27].y], 
         [pose.iloc[24].y, pose.iloc[23].y]
        ]
    return x, y

def animation_frame(f, sign, xmin, xmax, ymin, ymax, ax):
    frame = sign[sign.frame==f]
    left = frame[frame.type=='left_hand']
    right = frame[frame.type=='right_hand']
    pose = frame[frame.type=='pose']
    face = frame[frame.type=='face'][['x', 'y']].values
    lx, ly = get_hand_points(left)
    rx, ry = get_hand_points(right)
    px, py = get_pose_points(pose)
    ax.clear()
    ax.plot(face[:,0], face[:,1], '.')
    for i in range(len(lx)):
        ax.plot(lx[i], ly[i])
    for i in range(len(rx)):
        ax.plot(rx[i], ry[i])
    for i in range(len(px)):
        ax.plot(px[i], py[i])
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    
@app.route('/', endpoint="index")
def index():
    return render_template("index.html")

@app.route('/menu')
def menu():
    # Your menu route logic goes here
    return render_template('menu.html')

@app.route('/upload', methods=['POST', 'GET'])
def upload():
    global file_date

    if request.method == 'POST':

        # Get the fileName from the form data
        fileName = request.form.get('fileName')

        now = datetime.datetime.now()
        # Format the date and time as "ddmmyyyy hh:mm:ss"
        now = now.strftime("%Y-%m-%d %Hh%Mm%Ss")
        file_date = now

        # Record the start time
        start_time = time.time()

        print(f'static/{fileName}')

        # get the landmark of face, pose and hand
        all_landmarks = do_capture_loop(f'static/{fileName}')

        # Record the end time
        end_time = time.time()

        # Calculate the elapsed time
        elapsed_time = end_time - start_time

        print(elapsed_time)
        print('D:/NEW/output/file/vid_{}.parquet'.format(str(file_date).replace(":", '')))
        # saves the landmark of sign into parquet
        all_landmarks.to_parquet(
            'D:/NEW/output/file/vid_{}.parquet'.format(str(file_date).replace(":", '')))

        return redirect(url_for('get_predicted_result'))
    elif request.method == 'GET':
        submit_button_value = request.args.get('submit_button')
        if submit_button_value == 'webcam':
            return render_template("index.html")
        elif submit_button_value == 'video_upload':
            return render_template('videoUpload.html')

@app.route('/text_to_sign', methods=['POST', 'GET'])
def text_to_sign():
    return render_template('textToSign.html')

@app.route('/translate_btn', methods=['POST', 'GET'])
def translate_btn():
    if request.method == 'POST':

        global tut_video_path, content

        data = request.get_json()
        content = data.get('content', '')

        if content in train['sign'].values:
            # Get the path from the 'path' column
            content = content.lower()
            tut_video_path = train.loc[train['sign'] == content, 'video_path'].values[0]
            print(f"Path for content '{content}': {tut_video_path}")

            # You can send a response back to the client if needed
            return jsonify({'message': tut_video_path})

    return render_template('textToSign.html')

@app.route('/video_skeletion', methods=['POST', 'GET'])
def video_skeletion():
    global content

    if request.method == 'POST':
        selected_option = request.form.get('selectedOption')
        print(selected_option)
        if selected_option == 'Video':
            if content in train['sign'].values:
                # Get the path from the 'path' column
                tut_video_path = train.loc[train['sign'] == content, 'video_path'].values[0]
                print(tut_video_path)
                # You can send a response back to the client if needed
                return jsonify({'message': tut_video_path})
        
        elif selected_option == 'Skeleton Pose':

            if content in train['sign'].values:
                # Get the path from the 'path' column
                skeleton_path = train.loc[train['sign'] == content, 'skeleton_path'].values[0]

                sign_parquet = pd.read_parquet(skeleton_path)
                sign_parquet.y = sign_parquet.y * -1

                fig, ax = plt.subplots()
                l, = ax.plot([], [])

                ## These values set the limits on the graph to stabilize the video
                xmin = sign_parquet.x.min() - 0.2
                xmax = sign_parquet.x.max() + 0.2
                ymin = sign_parquet.y.min() - 0.2
                ymax = sign_parquet.y.max() + 0.2
                
                animation = FuncAnimation(fig, func=animation_frame, frames=sign_parquet.frame.unique(), \
                                        fargs=(sign_parquet, xmin, xmax, ymin, ymax, ax))

                animation.save('static/tut_skeleton/output.mp4', writer='ffmpeg', fps=10)

                return jsonify({'message': 'tut_skeleton/output.mp4'})

    return render_template('textToSign.html')


@app.route('/get_predicted_result')
def get_predicted_result():
    pq_file = 'D:/NEW/output/file/vid_{}.parquet'.format(
        str(file_date).replace(":", ''))
    top_n_signs = get_prediction(pq_file, prediction_fn)
    return jsonify({'prediction': top_n_signs})


@app.route('/textpreprocessing')
def textpreprocessing():
    input_text = request.args.get('input_text', default='', type=str)

    # Define a translation table to replace punctuation with spaces
    translator = str.maketrans(
        string.punctuation, ' ' * len(string.punctuation))

    # Use translate to remove punctuation
    input_text = input_text.translate(translator)

    # Remove double whitespaces
    input_text = re.sub(r'\s+', ' ', input_text)

    inputs = tokenizer.encode("punctuate: " + input_text, return_tensors="tf")
    result = model.generate(inputs)

    decoded_output = tokenizer.decode(result[0], skip_special_tokens=True)

    return jsonify({'processed_text': decoded_output})


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/requests', methods=['POST', 'GET'])
def tasks():
    global cap, frameNo, all_landmarks, file_date, file_path

    if request.method == 'POST':
        global camera

        print(request.form.get('rec'))
        print(request.form.get('camera-check    '))
        if request.form.get('rec') == 'Start/Stop Recording':
            global rec, out
            rec = not rec

            now = datetime.datetime.now()
            # Format the date and time as "ddmmyyyy hh:mm:ss"
            now = now.strftime("%Y-%m-%d %Hh%Mm%Ss")

            if (rec):
                fourcc = cv2.VideoWriter_fourcc(*'MP4V')

                file_date = now

                if not os.path.exists('D:/NEW/output/video'):
                    os.makedirs('D:/NEW/output/video')
                    print(f"Folder '{'D:/NEW/output/video'}' created successfully.")
                else:
                    print(f"Folder '{'D:/NEW/output/video'}' already exists.")

                file_path = 'D:/NEW/output/video/vid_{}.mp4'.format(
                    str(file_date).replace(":", ''))

                out = cv2.VideoWriter(file_path, fourcc, 15.0, (640, 480))
                # Start new thread for recording the video concurrently at record function
                thread = Thread(target=record, args=[out,])
                thread.start()
            elif (rec == False):
                # stop recording and save the final video
                out.release()

                # Record the start time
                start_time = time.time()

                # get the landmark of face, pose and hand
                all_landmarks = do_capture_loop(file_path)

                # Record the end time
                end_time = time.time()

                # Calculate the elapsed time
                elapsed_time = end_time - start_time

                print(elapsed_time)

                if not os.path.exists('D:/NEW/output/file'):
                    os.makedirs('D:/NEW/output/file')
                    print(f"Folder '{'D:/NEW/output/file'}' created successfully.")
                else:
                    print(f"Folder '{'D:/NEW/output/file'}' already exists.")

                # saves the landmark of sign into parquet
                all_landmarks.to_parquet(
                    'D:/NEW/output/file/vid_{}.parquet'.format(str(file_date).replace(":", '')))
        elif request.form.get('camera-check') == 'open':
            camera = not camera

        elif request.form.get('camera-check') == None:
            camera = not camera

    elif request.method == 'GET':
        return render_template('index.html')
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)

cap.release()
cv2.destroyAllWindows()
