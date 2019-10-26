"""Shopper Gaze Monitor."""

"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os, os.path
import sys
import json
import time
import cv2
import numpy as np
# from cbhelpers import log-helper as logger

import logging as log
# import paho.mqtt.client as mqtt

from threading import Thread
from collections import namedtuple
from argparse import ArgumentParser
from inference import Network
from loghelper import LogHelper as lh

# Environment set up
MyStruct = namedtuple("shoppingInfo", "faces, lookers, attenders, totalSessions, session")
INFO = MyStruct(0, 0, 0, 0, False) # shoppingInfo contains statistics for the shopping information
MyPath = namedtuple("MyPath", "FACE_DETECTION, HEAD_POSITION, FACE_REID, CPU_EXTENSION_PATH, FACE_DIR, OUTPUT_DIR")
def initializePATH():  # setup and return path struct
    # paths to the various trained models
    MODEL_PATH_ROOT = "C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\\tools\model_downloader"
    FACE_DETECTION = f"{MODEL_PATH_ROOT}\\Transportation\\object_detection\\face\\pruned_mobilenet_reduced_ssd_shared_weights\\dldt\\INT8\\face-detection-adas-0001.xml"
    HEAD_POSITION = f"{MODEL_PATH_ROOT}\\Transportation\\object_attributes\\headpose\\vanilla_cnn\\dldt\\INT8\\head-pose-estimation-adas-0001.xml"
    FACE_REID = f"{MODEL_PATH_ROOT}\\Retail\\object_reidentification\\face\\mobilenet_based\\dldt\\FP32\\face-reidentification-retail-0095.xml"
    # path to device driver
    CPU_EXTENSION_PATH = "c:\Program Files (x86)\IntelSWTools\openvino\inference_engine\lib\intel64\Release\inference_engine.lib"
    # paths to image dirs
    FACE_DIR = ".\\dataset\\faces"
    OUTPUT_DIR = ".\\output"
    # MyPath = namedtuple("MyPath", "FACE_DETECTION, HEAD_POSITION, FACE_REID, CPU_EXTENSION_PATH, FACE_DIR, OUTPUT_DIR")
    return MyPath(FACE_DETECTION, HEAD_POSITION, FACE_REID, CPU_EXTENSION_PATH, FACE_DIR, OUTPUT_DIR)
PATH = initializePATH() # PATH contatins directory paths to various resources
def intializeMQTT():  # Set up and return MQTT global variables
    TOPIC = "shopper_gaze_monitor"
    MQTT_HOST = "localhost"
    MQTT_PORT = 1883
    MQTT_KEEPALIVE_INTERVAL = 60
    return (TOPIC, MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
#TOPIC, MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL = intializeMQTT(): # Set up MQTT server environment variables
MyState = namedtuple("MyState", "KEEP_RUNNING, DELAY, FRAMES_SINCE_LAST_LOG, FRAMES_TO_WAIT_BETWEEN_LOGS, CURRENT_SESSION_FACE, CURRENT_SESSION_FACEIMAGE")
# paths to image dirs
def intializeSTATE():
    KEEP_RUNNING = True
    DELAY = 5
    FRAMES_SINCE_LAST_LOG = 0
    FRAMES_TO_WAIT_BETWEEN_LOGS = 100
    CURRENT_SESSION_FACE = []
    CURRENT_SESSION_FACEIMAGE = []
    return MyState(KEEP_RUNNING, DELAY, FRAMES_SINCE_LAST_LOG, FRAMES_TO_WAIT_BETWEEN_LOGS, CURRENT_SESSION_FACE, CURRENT_SESSION_FACEIMAGE)
STATE = intializeSTATE() # STATE holds vars to control background thread
logger = log.getLogger() 

def args_parser():
    """
    Parse command line arguments.
    :return: Command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=False,
                        help="Path to an .xml file with a pre-trained"
                        "face detection model")
    parser.add_argument("-pm", "--posemodel", required=False,
                        help="Path to an .xml file with a pre-trained model"
                        "head pose model")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to video file or image."
                        "'cam' for capturing video stream from camera")
    parser.add_argument("-l", "--cpu_extension", type=str, default=None,
                        help="MKLDNN (CPU)-targeted custom layers. Absolute "
                        "path to a shared library with the kernels impl.")
    parser.add_argument("-d", "--device", default="CPU", type=str,
                        help="Specify the target device to infer on; "
                        "CPU, GPU, FPGA, HDDL or MYRIAD is acceptable. Application "
                        "will look for a suitable plugin for device specified"
                        "(CPU by default)")
    parser.add_argument("-c", "--confidence", default=0.5, type=float,
                        help="Probability threshold for detections filtering")

    return parser
def initializeInput(input):
    """
    Takes in script args.input, an routes it to an appropriate processing function.
    :param input: Int (indicating camera to take stream from), file, or dir .  Script args.input.
    :return: Video capture stream (vcap), and frames per sec delay.
    """
    lh.logfunc(initializeInput, locals())
    if input == 'cam':
        return initializeVidStreamInput(input)
    elif input == 'dir':
        return initializeDirOfImagesInput(input)
    else:
        return initializeImageFileInput(input)
def initializeVidStreamInput(input):
    lh.logfunc(initializeVidStreamInput, locals())
    print(f">> initializeVidStreamInput({type(input)} {input});")
    input_stream = 0
    print(f"...initializing camera stream ({type(input_stream)} {input_stream})")
    try:
        vcap = cv2.VideoCapture(input_stream)   
        vcap.open(input_stream)
        print(f"...vcap.open({type(input_stream)} {input_stream})")
        # Adjust DELAY to match the number of FPS of the video file
        delay = 1000 / vcap.get(cv2.CAP_PROP_FPS)
        assert vcap.isOpened(), "ERROR! Unable to open video source"
        return vcap,delay
    except AssertionError as e:
        logger.error(e)
        print(e)
        return None, None
    except Exception as e: 
        logger.error(e)
        print(e)
        return None, None
def initializeDirOfImagesInput(input):
    lh.logfunc(initializeDirOfImagesInput, locals())
    try:
        print(f"> initializeDirOfImagesInput({type(input)} {input})")    
        assert os.path.isfile(input), "Specified input file doesn't exist"
        input_stream = input
        delay = 3000
        return input_stream, delay
    except AssertionError as e:
        logger.error(e)
        print(e)
        return None, None
    except Exception as e: 
        logger.error(e)
        print(e)
        return None, None
def initializeImageFileInput(input):
    lh.logfunc(initializeImageFileInput, locals())
    try:
        print(f"> initializeImageFileInput({type(input)} {input})")
        assert os.path.isfile(input), "Specified input file doesn't exist"
        path = os.path.abspath("mydir/myfile.txt")
        flag = cv2.IMREAD_COLOR
        input_stream = cv2.imread(path, flag)
        delay = 3000
        return input_stream, delay
    except AssertionError as e:
        logger.error(e)
        print(e)
        return None, None
    except Exception as e: 
        logger.error(e)
        print(e)
        return None, None
def message_runner():
    """
    Publish worker status to MQTT topic.
    Pauses for rate second(s) between updates
    :return: None
    """
    global STATE
    while STATE.KEEP_RUNNING:
        payload = json.dumps({"Faces": INFO.faces, "Lookers": INFO.lookers, "Attenders": INFO.attenders, "totalSessions": INFO.totalSessions, "session": INFO.session})
        time.sleep(1)
        CLIENT.publish(TOPIC, payload=payload)

def getModelOutput(name,model,device,input_size,output_size,num_requests,cpu_extension,plugin=None):
        """
         Loads a network and an image to the Inference Engine plugin.
        :param model: .xml file of pre trained model
        :param cpu_extension: extension for the CPU device
        :param device: Target device
        :param input_size: Number of input layers
        :param output_size: Number of output layers
        :param num_requests: Index of Infer request value. Limited to device capabilities.
        :param plugin: Plugin for specified device
        :return:  Shape of input layer
        """
        lh.logfunc(getModelOutput, locals())
        initializeImageFileInput
        infer_network = Network()
        res = {
            'name':name, 'model':model, 'device':device, 'network':infer_network,
            'input_size':input_size, 'output_size':output_size, 'num_requests':num_requests,
            'cpu_extension':cpu_extension
        }
        res['plugin'], (res['n'],res['c'],res['h'],res['w']) = infer_network.load_model(
                                            model,  # Head position.  n,c,heigth,width?
                                            device, 
                                            input_size,
                                            output_size,
                                            num_requests,
                                            cpu_extension, 
                                            plugin)  #TODO figure out what this plugin param does.
        print(f"...inference results for {name}: {res}")
        return res
def getFaceImage(face,frame):
        xmin, ymin, xmax, ymax = face
        faceImage = frame[ymin:ymax, xmin:xmax]
        return faceImage
def getImageHW(img):
    height, width, channels = img.shape
    return {'h':height, 'w':width, 'channels':channels}
def isImage(faceImage):
    try:
        shape = faceImage.shape
        return True
    except:
        return False
def face_detection(vcap,next_frame,fd): # parse faces from an image
    """
    Parse Face detection output.
    :param res: Detection results
    :param args: Parsed arguments
    :param initial_wh: Initial width and height of the FRAME
    :return: Co-ordinates of the detected face
    """
    global INFO
    global STATE
    global args
    faces,det_time_fd = [],0
    # Get intial width and height of video stream
    initial_wh = [vcap.get(3), vcap.get(4)]
    in_frame_fd = cv2.resize(next_frame, (fd['w'], fd['h']))
    # Change data layout from HWC to CHW
    in_frame_fd = in_frame_fd.transpose((2, 0, 1))
    in_frame_fd = in_frame_fd.reshape((fd['n'], fd['c'], fd['h'], fd['w']))
    key_pressed = cv2.waitKey(int(STATE.DELAY))

    # Start asynchronous inference for specified request
    inf_start_fd = time.time()
    fd['network'].exec_net(0, in_frame_fd)

    # Wait for the result
    fd['network'].wait(0)
    det_time_fd = time.time() - inf_start_fd

    # Results of the output layer of the network
    res = fd['network'].get_output(0) #plugin

    for obj in res[0][0]:
        # Draw only objects when probability more than specified threshold
        if obj[2] > args.confidence:
            if obj[3] < 0:
                obj[3] = -obj[3]
            if obj[4] < 0:
                obj[4] = -obj[4]
            xmin = int(obj[3] * initial_wh[0])
            ymin = int(obj[4] * initial_wh[1])
            xmax = int(obj[5] * initial_wh[0])
            ymax = int(obj[6] * initial_wh[1])
            faces.append([xmin, ymin, xmax, ymax])
    return faces,det_time_fd
def parseLookers(faces,frame,hp): # Parse faces for faces that are looking
    lookingFaces,det_time_hp = [],0
    if faces:
        for face in faces:
            xmin, ymin, xmax, ymax = face
            head_pose = frame[ymin:ymax, xmin:xmax]
            in_frame_hp = cv2.resize(head_pose, (hp['w'], hp['h']))
            in_frame_hp = in_frame_hp.transpose((2, 0, 1))
            in_frame_hp = in_frame_hp.reshape((hp['n'], hp['c'], hp['h'], hp['w']))
            inf_start_hp = time.time()
            hp['network'].exec_net(0, in_frame_hp)
            hp['network'].wait(0)
            det_time_hp = time.time() - inf_start_hp

            # Parse head pose detection results
            angle_p_fc = hp['network'].get_output(0, "angle_p_fc")
            angle_y_fc = hp['network'].get_output(0, "angle_y_fc")
            if ((angle_y_fc > -22.5) & (angle_y_fc < 22.5) & (angle_p_fc > -22.5) &
                    (angle_p_fc < 22.5)):
                lookingFaces.append(face)
    return lookingFaces,det_time_hp
def parseAttenders(lookingFaces,frame,vcap,initial_wh,x_criteria,y_criteria): # Parse looking faces for faces that 
    attendingFaces = []
    frame_wh = [vcap.get(3), vcap.get(4)]
    assert (frame_wh == initial_wh), "Error! frame_wh [{0}] != initial_wh [{1}]".format(frame_wh,initial_wh)
    if lookingFaces:
        for face in lookingFaces:
            xmin, ymin, xmax, ymax = face
            if (xmax-xmin > (frame_wh[0]*x_criteria) or ymax-ymin > (frame_wh[1]*y_criteria)):
                attendingFaces.append(face)
    return attendingFaces
def getFaceEmbedding(faceimage,fe):
    """
    Convert face image to face embedding.
    :param faceImage: Cropped image of face
    :param ri: Face Reidentification model
    :return: Facial embedding
    """
    faceEmbedding,det_time_fe = [],0
    global STATE

    if faceimage.size > 0:  #generate face embedding from image

        # Get intial width and height of video stream
        initial_wh = getImageHW(faceimage)
        in_frame_fe = cv2.resize(faceimage, (fe['w'], fe['h']))
        # Change data layout from HWC to CHW
        in_frame_fe = in_frame_fe.transpose((2, 0, 1))
        in_frame_fe = in_frame_fe.reshape((fe['n'], fe['c'], fe['h'], fe['w']))
        key_pressed = cv2.waitKey(int(STATE.DELAY))

        # Start asynchronous inference for specified request
        inf_start_fe = time.time()
        fe['network'].exec_net(0, in_frame_fe)

        # Wait for the result
        fe['network'].wait(0)
        det_time_fe = time.time() - inf_start_fe

        # Results of the output layer of the network
        faceEmbedding = fe['network'].get_output(0) #plugin

    return faceEmbedding,det_time_fe

def draw_results(frame,faces,lookingFaces,attendingFaces,det_time_fd,det_time_hp):
    """
    Parse SSD output.

    :param frame: frame from camera/video
    :param result: list contains the data to parse ssd
    :return: person count and frame
    """
    # Define on screen messages
    stats = {
        'texts': [
            "Face Inference time: {:.3f} ms.".format(det_time_fd * 1000),
            "Faces: {}".format(INFO.faces),
            "Lookers: {}".format(INFO.lookers),
            "Attenders: {}".format(INFO.attenders),
            "Total Sessions: {}".format(INFO.totalSessions),
            "Session: {}".format(INFO.session)
        ]
    }
    styles = { 
        'text':{
            'originOrg': (10,20),
            'fontFace': cv2.FONT_HERSHEY_COMPLEX,
            'fontScale': 0.5,
            'color': (0,0,255),
            'thickness': 1,
            'lineStyle':8,
            'padding':20,
            'backgroundColor':(255,255,255),
            'borderThickness':-1,
        },
        'boxes':{
            'originOrg': (10,20),
            'fontFace': cv2.FONT_HERSHEY_COMPLEX,
            'fontScale': 0.5,
            'color': (0,255,0),
            'thickness': 3,
            'lineStyle':8,
        }
    }
    if det_time_hp:
        stats['texts'].insert(1,"Head pose Inference time: {:.3f} ms.".format(det_time_hp * 1000))
    
    # Draw on screen messages    
    frame = drawText(frame,stats,styles['text'])
    # Draw boxes
    drawFaces(frame,faces,lookingFaces,attendingFaces,styles['boxes'])
    # Draw embeddings
    #drawEmbeddings(frame,embeddings)
    # update frame
    cv2.imshow("Shopper Gaze Monitor", frame) # Draws above to screen
    return True
def drawText(frame,stats,styles):
    charHeight, charWidth= [20,10]
    th = charHeight * len(stats['texts'])
    tw = charWidth * len(max(stats['texts'], key=len))
    # draws the background box behind the msgs
    cv2.rectangle(
        frame, 
        tuple((p-styles['padding']) for p in styles['originOrg']), 
        tuple((p+styles['padding']) for p in (tw,th)), 
        styles['backgroundColor'], 
        styles['borderThickness'],
        ) 
    # draw text
    for i,text in enumerate(stats['texts']):
        org = (styles['originOrg'][0], styles['originOrg'][1]+i*charHeight)
        cv2.putText(frame, text, org, 
            styles['fontFace'], styles['fontScale'], styles['color'], 
            styles['thickness'], styles['lineStyle'])
    return frame
def drawFaces(frame,faces,lookingFaces,attendingFaces,styles):
    # Draw boxes around faces
    if attendingFaces:
        for attender in attendingFaces:
            xmin, ymin, xmax, ymax = attender      
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,255,0), 5)
    if faces:
        for face in faces:
            xmin, ymin, xmax, ymax = face      
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255,255,255), 1)
    if lookingFaces:
        for looker in lookingFaces:
            xmin, ymin, xmax, ymax = looker      
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255,0,0), 1)
    return frame
def drawEmbeddings(frame,embeddings):
    radius = 3
    color = (0,255,0)
    thickness = -1
    for e in embeddings:
        center_coordinates = (e[0],e[1])
        cv2.circle(frame, center_coordinates, radius, color, thickness)
    return frame
def logFrame(faces,lookingFaces,attendingFaces):
    global STATE
    global INFO
    # INFO("shoppingInfo", "faces, lookers, attenders, totalSessions, session")
    # STATE("MyState", "KEEP_RUNNING, DELAY, FRAMES_SINCE_LAST_LOG, FRAMES_TO_WAIT_BETWEEN_LOGS, CURRENT_SESSION_FACE, CURRENT_SESSION_FACEIMAGE")

    # update FRAMES_SINCE_LAST_LOG
    STATE = STATE._replace(FRAMES_SINCE_LAST_LOG = STATE.FRAMES_SINCE_LAST_LOG +1)

    # log frame
    if STATE.FRAMES_SINCE_LAST_LOG >= STATE.FRAMES_TO_WAIT_BETWEEN_LOGS:
        print("")
        print(".")
        print("...")
        print(".....")
        print(f"...logging session state after {STATE.FRAMES_SINCE_LAST_LOG} frames:")
        print(".")
        print(f"[{len(faces)==INFO.faces}] len(faces) {len(faces)} == INFO.faces {INFO.faces};")
        print(f"[{len(lookingFaces)==INFO.lookers}] len(lookingFaces) {len(lookingFaces)} == INFO.lookingFaces {INFO.lookers};; ")
        print(f"[{len(attendingFaces)==INFO.attenders}] len(attendingFaces) {len(attendingFaces)} == INFO.attendingFaces {INFO.attenders};")
        print(".")
        print(f"[{len(STATE.CURRENT_SESSION_FACE)}] len(STATE.CURRENT_SESSION_FACE);")
        showSessionFace(STATE.CURRENT_SESSION_FACEIMAGE)
        print(".....")
        print("...")
        print(".")
        STATE = STATE._replace(FRAMES_SINCE_LAST_LOG = 0)


def isNewFace(faceEmbedding):
    global STATE
    # Check provided face embedding against STATE.CURRENT_SESSION_FACE
    return faceEmbedding != STATE.CURRENT_SESSION_FACE
def updateSession(b):
    global INFO
    INFO = INFO._replace(session=b)
    if b:
        # print("INFO.totalSessions: {0} {1}; INFO.sessions: {2} {3}; ".format(type(INFO.totalSessions),INFO.totalSessions,type(INFO.sessions),INFO.sessions))
        INFO = INFO._replace(totalSessions=(INFO.totalSessions + 1))
        updateSessionFace(faceimage,faceEmbedding)
def updateSessionFace(faceEmbedding,faceimage):
    global STATE
    STATE = STATE._replace(
        CURRENT_SESSION_FACE=faceEmbedding,
        CURRENT_SESSION_FACEIMAGE=faceimage
    )
def showSessionFace(faceimage):
    #TODO: Write face to STATE.OUTPUT_DIR
    if isImage(faceimage):
        print(f"...showing session face.")
        cv2.imwrite("SessionFace.jpg",faceimage)
        wn = "Session Face"
        cv2.namedWindow(wn)
        cv2.moveWindow(wn,700,200)
        cv2.imshow(wn,faceimage)
    else:
        print(f"...no current session face.")
def needNewSession(session,frame,face,frame_wh):
    res = False
    if session and not attending: # check if session is abandoned (no face)
        print(f"...session:{session}, attending:{attending}.  Clearing session.")
        updateSessionFace(None)
        res = True
    elif not session and attending:
        print(f"...session:{session}, attending:{attending}.  Starting new session.")
        res = True
    elif not session and not attending:  # do nothing
        res = False
    elif session and attending:  # start a new session
        res = False
    else:
        pass
    return res
def processFace(faceimage):
    print(f"...processing face")
    return False
    global FACE_DIR
    global OUTPUT_DIR
    faceEmbedding = processFace(faceimage)
    showSessionFace(faceimage)
    updateSessionFace(faceimage,faceEmbedding)
    if isNewFace(faceEmbedding):
        print(f"...new face. Starting new session.")
        facecount = len([name for name in os.listdir(OUTPUT_DIR) if os.path.isfile(name)])
        if facecount < 10: # TODO: Remove this  when face embeddings works
            cv2.imwrite(f"{OUTPUT_DIR}/face_{facecount}.jpg", faceimage)
            print(f"...writing new face to: \"{OUTPUT_DIR}/face_{facecount}.jpg\"")
        else:
            print(f"...too many faces!")
        res = True
    else:
        print(f"WARNING: unhandled case in ")

def processFrame(vcap,next_frame,fd,hp,fe): # process a frame of a vcap
    global INFO
    global STATE
    global args
    det_time_hp = None
    frame = next_frame
    initial_wh = [vcap.get(3), vcap.get(4)]
    key_pressed = cv2.waitKey(int(STATE.DELAY))
    # Parse face detection output
    faces,det_time_fd = face_detection(vcap,next_frame,fd)

    # Look for faces that are looking at the camera
    lookingFaces,det_time_hp = parseLookers(faces,frame,hp)
    
    # Check for attending faces (looker is more than 75% of the camera h or w)
    x_criteria = 0.25
    y_criteria = 0.50
    attendingFaces = parseAttenders(lookingFaces,frame,vcap,initial_wh,x_criteria,y_criteria)
    
    needNewSession = True
    if(attendingFaces):
        fes = []
        for face in attendingFaces:
            faceImage = getFaceImage(face,frame)
            fes.append( {'embedding':getFaceEmbedding(faceImage,fe), 'image':faceImage} )
        for fe in fes:
            # if STATE.CURRENT_SESSION_FACE != fe['embedding']:
            if True: #TODO: Need a function to return cosine similarity of 2 embeddings
                needNewSession == False
                print(f"...len(fe['embedding']):{len(fe['embedding'])}")
                print(f"...len(fe['image']):{len(fe['image'])}")
                # print(fe)

                showSessionFace(fe['image'])
                break
            else:
                updateSessionFace(fe['embedding'],fe['image'])
                pass
        if needNewSession:
            pass

        

    # Check whether a new session should be started
    #session = needNewSession(INFO.session,frame,face,frame_wh)
    # updateSession(session)

    #update INFO
    INFO = INFO._replace(
        faces=len(faces),
        lookers=len(lookingFaces),
        attenders=len(attendingFaces)
    )
    # Draw performance stats
    draw_results(frame,faces,lookingFaces,attendingFaces,det_time_fd,det_time_hp)    
    # log frames stats
    logFrame(faces,lookingFaces,attendingFaces)
    return next_frame, key_pressed

def initilizeModels(device,paths):
    print(f"...initializing models")
    fdmodel = {
        'name':'fd',
        'model':paths.FACE_DETECTION, 
        'device':device,
        'input_size':1, 'output_size':1, 'num_requests':0,
        'cpu_extension':paths.CPU_EXTENSION_PATH,
        'plugin':None
    }
    print(f"...initializing {fdmodel['name']} with model: {fdmodel['model']}")
    fd = getModelOutput(**fdmodel) # {name,model,device,network,up1,up2,up3,cpu_extension,plugin,n,c,h,w,}

    hpmodel = {
        'name':'hp',
        'model':paths.HEAD_POSITION,
        'device':device,
        'input_size':1, 'output_size':3, 'num_requests':0,
        'cpu_extension':paths.CPU_EXTENSION_PATH,
        'plugin':fd['plugin']
    } 
    print(f"...initializing {hpmodel['name']} with model: {hpmodel['model']}")
    hp = getModelOutput(**hpmodel) # {name,model,device,network,up1,up2,up3,cpu_extension,plugin,n,c,h,w,}

    rimodel = {
        'name':'fe',
        'model':paths.FACE_REID,
        'device':device,
        'input_size':1, 'output_size':1, 'num_requests':0,
        'cpu_extension':paths.CPU_EXTENSION_PATH,
        'plugin':None
    } 
    print(f"...initializing {rimodel['name']} with model: {rimodel['model']}")
    ri = getModelOutput(**rimodel) # {name,model,device,network,up1,up2,up3,cpu_extension,plugin,n,c,h,w,}

    return (fd,hp,ri)

def main():
    """
    Load the network and parse the output.
    :return: None
    """
    # Setup and parse args
    global INFO
    global DELAY
    global CLIENT
    global POSE_CHECKED 
    global args
    global PATH
    global STATE
    # CLIENT = mqtt.Client()
    # CLIENT.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    # log.basicConfig(format="[ %(levelname)s ] %(message)s",
    #                 level=log.INFO, stream=sys.stdout)
    args = args_parser().parse_args()
    # if video stream, Establish video stream
    vcap, delay = initializeInput(args.input)
    STATE = STATE._replace(DELAY=delay)
    ret, frame = vcap.read()
    
    # Initialize the class
    fd,hp,fe = initilizeModels(args.device,PATH)

    # mqtt stuff
    # message_thread = Thread(target=message_runner)
    # message_thread.setDaemon(True)
    # message_thread.start()
    
    INFO = INFO._replace(session=False)

    # Main vid loop.  Repeats for each frame
    while ret:
        
        ret, next_frame = vcap.read()
        
        if not ret:
            STATE = STATE._replace(KEEP_RUNNING=False)
            break

        if next_frame is None:
            STATE = STATE._replace(KEEP_RUNNING=False)
            log.error("ERROR! blank FRAME grabbed")
            break  

        next_frame,key_pressed = processFrame(vcap,next_frame,fd,hp,fe)

        if key_pressed == 27:
            print("Attempting to stop background threads")
            STATE = STATE._replace(KEEP_RUNNING=False)
            break

    fd['network'].clean()
    hp['network'].clean()
    ri['network'].clean()
    # message_thread.join()
    vcap.release()
    cv2.destroyAllWindows()
    # CLIENT.disconnect()

if __name__ == '__main__':
    main()
    sys.exit()