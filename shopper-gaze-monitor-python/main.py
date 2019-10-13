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

import os
import sys
import json
import time
import cv2
# from cbhelpers import log-helper as logger

import logging as log
# import paho.mqtt.client as mqtt

from threading import Thread
from collections import namedtuple
from argparse import ArgumentParser
from inference import Network

# shoppingInfo contains statistics for the shopping information
MyStruct = namedtuple("shoppingInfo", "shopper, looker, session, totalSessions")
INFO = MyStruct(0, 0, False, 0)

POSE_CHECKED = False

# MQTT server environment variables
TOPIC = "shopper_gaze_monitor"
# MQTT_HOST = "localhost"
# MQTT_PORT = 1883
# MQTT_KEEPALIVE_INTERVAL = 60

# Flag to control background thread
KEEP_RUNNING = True

DELAY = 5

logger = log.getLogger() 

def args_parser():
    """
    Parse command line arguments.
    :return: Command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True,
                        help="Path to an .xml file with a pre-trained"
                        "face detection model")
    parser.add_argument("-pm", "--posemodel", required=True,
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
    print(f">> initializeInput({type(input)} {input})")
    if input == 'cam':
        return initializeVidStreamInput(input)
    elif input == 'dir':
        return initializeDirOfImagesInput(input)
    else:
        return initializeImageFileInput(input)
def initializeVidStreamInput(input):
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
def face_detection(res, args, initial_wh):
    """
    Parse Face detection output.
    :param res: Detection results
    :param args: Parsed arguments
    :param initial_wh: Initial width and height of the FRAME
    :return: Co-ordinates of the detected face
    """
    global INFO
    faces = []
    INFO = INFO._replace(shopper=0)

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
            INFO = INFO._replace(shopper=len(faces))
    return faces
def message_runner():
    """
    Publish worker status to MQTT topic.
    Pauses for rate second(s) between updates
    :return: None
    """
    while KEEP_RUNNING:
        payload = json.dumps({"Shopper": INFO.shopper, "Looker": INFO.looker, "Session": INFO.session, "totalSessions": INFO.totalSessions})
        time.sleep(1)
        CLIENT.publish(TOPIC, payload=payload)
def draw_results(frame, faces, det_time_fd, det_time_hp):
    """
    Parse SSD output.

    :param frame: frame from camera/video
    :param result: list contains the data to parse ssd
    :return: person count and frame
    """
    # Define on screen messages
    stats = {
        'style':{
            'originOrg': (10,20),
            'fontFace': cv2.FONT_HERSHEY_COMPLEX,
            'fontScale': 0.5,
            'color': (0,0,255),
            'thickness': 1,
            'lineStyle':8,
        },
        'texts': [
            "Face Inference time: {:.3f} ms.".format(det_time_fd * 1000),
            "Shopper: {}".format(INFO.shopper),
            "Looker: {}".format(INFO.looker),
            "Sessions: {}".format(INFO.session)
        ]
    }
    if det_time_hp:
        stats['texts'].insert(1,"Head pose Inference time: {:.3f} ms.".format(det_time_hp * 1000))
    charHeight, charWidth= [20,10]
    originOrg, fontFace, fontScale, color, thickness, lineType = stats['style'].values()
    th = charHeight * len(stats['texts'])
    tw = charWidth * len(max(stats['texts'], key=len))

    # draws the background box behind the msgs
    padding = 20
    cv2.rectangle(
        frame, 
        tuple((p-padding) for p in stats['style']['originOrg']), 
        tuple((p+padding) for p in (tw,th)), 
        (255,255,255), 
        -1
        ) 
    # Draw on screen messages
    for i,text in enumerate(stats['texts']):
        org = (originOrg[0], originOrg[1]+i*charHeight)
        cv2.putText(frame, text, org, fontFace, fontScale, color, thickness, lineType)
    
    # Draw boxes around faces
    for res_hp in faces:
        xmin, ymin, xmax, ymax = res_hp      
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 255, 255), 1)

    cv2.imshow("Shopper Gaze Monitor", frame) # Draws above to screen

    return True
def getModelOutput(name,model,device,up1,up2,up3,cpu_extension,plugin=None):
        # TODO: Figure out what up1, up2, and up3 are/do.  (unknown parameters)
        print(f">> getModelOutput({name},{model},{device},{up1},{up2},{up3},{cpu_extension},{plugin})")
        infer_network = Network()
        res = {
            'name':name, 'model':model, 'device':device, 'network':infer_network,
            'up1':up1, 'up2':up2, 'up3':up3, 'cpu_extension':cpu_extension
        }
        res['plugin'], (res['n'],res['c'],res['h'],res['w']) = infer_network.load_model(
                                            model,  # Head position.  n,c,heigth,width?
                                            device, 
                                            up1,up2,up3,
                                            cpu_extension, 
                                            plugin)  #TODO figure out what this plugin param does.  Comes from line 229
        print(f"...inference results for {name}: {res}")
        return res
        # return res['network'], res['plugin'], (res['n'],res['c'],res['h'],res['w'])
def processFrame(vcap,fd,hp,ret,next_frame,faces,args):
    looking = 0
    session = False
    global INFO
    global DELAY
    frame = next_frame
    # Get intial width and height of video stream
    initial_wh = [vcap.get(3), vcap.get(4)]
    in_frame_fd = cv2.resize(next_frame, (fd['w'], fd['h']))
    # Change data layout from HWC to CHW
    in_frame_fd = in_frame_fd.transpose((2, 0, 1))
    in_frame_fd = in_frame_fd.reshape((fd['n'], fd['c'], fd['h'], fd['w']))
    key_pressed = cv2.waitKey(int(DELAY))

    # Start asynchronous inference for specified request
    inf_start_fd = time.time()
    fd['network'].exec_net(0, in_frame_fd)

    # Wait for the result
    fd['network'].wait(0)
    det_time_fd = time.time() - inf_start_fd

    # Results of the output layer of the network
    res = fd['network'].get_output(0)

    # Parse face detection output
    faces = face_detection(res, args, initial_wh)
    #check 
    if len(faces) != 0:
        # Look for poses
        for res_hp in faces:
            xmin, ymin, xmax, ymax = res_hp
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
                looking += 1
                POSE_CHECKED = True
                INFO = INFO._replace(looker=looking)
                # Check if looker is more than 75% of the camera h or w
                frame_wh = [vcap.get(3), vcap.get(4)]
                assert (frame_wh == initial_wh), "Error! frame_wh [{0}] != initial_wh [{1}]".format(frame_wh,initial_wh)

                if xmax-xmin > (frame_wh[0]*0.50) or ymax-ymin > (frame_wh[1]*0.50):
                    session = True
                    INFO = INFO._replace(session=True)
                else:
                    INFO = INFO._replace(session=False)
            else:
                INFO = INFO._replace(looker=looking)
    else:
        INFO = INFO._replace(looker=0)
        det_time_hp = None
    
    #Update Total Sessions
    if session:
        # print("INFO.totalSessions: {0} {1}; INFO.sessions: {2} {3}; ".format(type(INFO.totalSessions),INFO.totalSessions,type(INFO.sessions),INFO.sessions))
        INFO = INFO._replace(totalSessions=(INFO.totalSessions + 1))
    # Draw performance stats
    draw_results(frame, faces, det_time_fd, det_time_hp)
    # Draw performance stats
    draw_results(frame, faces, det_time_fd, det_time_hp)

    return next_frame, faces, key_pressed
def main():
    """
    Load the network and parse the output.
    :return: None
    """
    # Setup and parse args
    global INFO
    global DELAY
    global CLIENT
    global KEEP_RUNNING
    global POSE_CHECKED 
    # CLIENT = mqtt.Client()
    # CLIENT.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    # log.basicConfig(format="[ %(levelname)s ] %(message)s",
    #                 level=log.INFO, stream=sys.stdout)
    args = args_parser().parse_args()
    # if video stream, Establish video stream
    vcap, DELAY = initializeInput(args.input)
    ret, frame = vcap.read()
    # Initialize the class
    fdmodel = {
        'name':'fd',
        'model':args.model, 
        'device':args.device,
        'up1':1, 'up2':1, 'up3':0,
        'cpu_extension':args.cpu_extension,
        'plugin':None
    }
    fd = getModelOutput(**fdmodel) # {name,model,device,network,up1,up2,up3,cpu_extension,plugin,n,c,h,w,}
    hpmodel = {
        'name':'hp',
        'model':args.posemodel,
        'device':args.device,
        'up1':1, 'up2':3, 'up3':0,
        'cpu_extension':args.cpu_extension,
        'plugin':fd['plugin']
    } 
    hp = getModelOutput(**hpmodel) # {name,model,device,network,up1,up2,up3,cpu_extension,plugin,n,c,h,w,}

    # mqtt stuff
    # message_thread = Thread(target=message_runner)
    # message_thread.setDaemon(True)
    # message_thread.start()
    
    # Main vid loop.  Repeats for each frame
    while ret:
        ret, next_frame = vcap.read()
        if not ret:
            KEEP_RUNNING = False
            break
        if next_frame is None:
            KEEP_RUNNING = False
            log.error("ERROR! blank FRAME grabbed")
            break  
        faces = None
        next_frame,faces,key_pressed = processFrame(vcap,fd,hp,ret,next_frame,faces,args)

        # looking = 0
        # session = False
        # ret, next_frame = vcap.read()
        # # Get intial width and height of video stream
        # initial_wh = [vcap.get(3), vcap.get(4)]
        # in_frame_fd = cv2.resize(next_frame, (fd['w'], fd['h']))
        # # Change data layout from HWC to CHW
        # in_frame_fd = in_frame_fd.transpose((2, 0, 1))
        # in_frame_fd = in_frame_fd.reshape((fd['n'], fd['c'], fd['h'], fd['w']))

        # key_pressed = cv2.waitKey(int(DELAY))

        # # Start asynchronous inference for specified request
        # inf_start_fd = time.time()
        # fd['network'].exec_net(0, in_frame_fd)


        # # Wait for the result
        # fd['network'].wait(0)
        # det_time_fd = time.time() - inf_start_fd

        # # Results of the output layer of the network
        # res = fd['network'].get_output(0)

        # # Parse face detection output
        # faces = face_detection(res, args, initial_wh)
        # #check 
        # if len(faces) != 0:
        #     # Look for poses
        #     for res_hp in faces:
        #         xmin, ymin, xmax, ymax = res_hp
        #         head_pose = frame[ymin:ymax, xmin:xmax]
        #         in_frame_hp = cv2.resize(head_pose, (hp['w'], hp['h']))
        #         in_frame_hp = in_frame_hp.transpose((2, 0, 1))
        #         in_frame_hp = in_frame_hp.reshape((hp['n'], hp['c'], hp['h'], hp['w']))

        #         inf_start_hp = time.time()
        #         hp['network'].exec_net(0, in_frame_hp)
        #         hp['network'].wait(0)
        #         det_time_hp = time.time() - inf_start_hp


        #         # Parse head pose detection results
        #         angle_p_fc = hp['network'].get_output(0, "angle_p_fc")
        #         angle_y_fc = hp['network'].get_output(0, "angle_y_fc")
        #         if ((angle_y_fc > -22.5) & (angle_y_fc < 22.5) & (angle_p_fc > -22.5) &
        #                 (angle_p_fc < 22.5)):
        #             looking += 1

        #             POSE_CHECKED = True
        #             INFO = INFO._replace(looker=looking)
        #             # Check if looker is more than 75% of the camera h or w
        #             frame_wh = [vcap.get(3), vcap.get(4)]
        #             assert (frame_wh == initial_wh), "Error! frame_wh [{0}] != initial_wh [{1}]".format(frame_wh,initial_wh)

        #             if xmax-xmin > (frame_wh[0]*0.50) or ymax-ymin > (frame_wh[1]*0.50):
        #                 session = True
        #                 INFO = INFO._replace(session=True)
        #             else:
        #                 INFO = INFO._replace(session=False)
        #         else:
        #             INFO = INFO._replace(looker=looking)
        # else:
        #     INFO = INFO._replace(looker=0)
        #     det_time_hp = None
        
        # #Update Total Sessions
        # if session:
        #     # print("INFO.totalSessions: {0} {1}; INFO.sessions: {2} {3}; ".format(type(INFO.totalSessions),INFO.totalSessions,type(INFO.sessions),INFO.sessions))
        #     INFO = INFO._replace(totalSessions=(INFO.totalSessions + 1))

        if key_pressed == 27:
            print("Attempting to stop background threads")
            KEEP_RUNNING = False
            break

        print(f"INFO: {INFO}")

    fd['network'].clean()
    hp['network'].clean()
    # message_thread.join()
    vcap.release()
    cv2.destroyAllWindows()
    # CLIENT.disconnect()



if __name__ == '__main__':
    main()
    sys.exit()
