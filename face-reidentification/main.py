import os, os.path
import sys
import json
import time
import cv2
import logging as log
from threading import Thread
from collections import namedtuple
from argparse import ArgumentParser
from inference import Network

# shoppingInfo contains statistics for the shopping information
MyStruct = namedtuple("shoppingInfo", "shopper, looker, session, totalSessions")
INFO = MyStruct(0, 0, False, 0)

# Flag to control background thread
KEEP_RUNNING = True
DELAY = 5
CURRENT_SESSION_FACE = None
CURRENT_SESSION_FACEIMAGE = None
logger = log.getLogger() 
FACE_DIR = ".\\dataset\\faces"
OUTPUT_DIR = ".\\output"
FACE_DETECTION_MODEL = "c:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\tools\model_downloader\Transportation\object_detection\face\pruned_mobilenet_reduced_ssd_shared_weights\dldt\INT8\face-detection-adas-0001.xml" 
HEAD_POSITION_MODEL = "c:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\tools\model_downloader\Transportation\object_attributes\headpose\vanilla_cnn\dldt\INT8\head-pose-estimation-adas-0001.xml" 
FACE_REID_MODEL = "c:\Program Files (x86)\IntelSWTools\openvino_2019.2.242\deployment_tools\open_model_zoo\tools\downloader\Retail\object_reidentification\face\mobilenet_based\dldt\FP16\face-reidentification-retail-0095.bin"
D = "GPU"
L = "c:\Program Files (x86)\IntelSWTools\openvino\inference_engine\lib\intel64\Release\inference_engine.lib"

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
        path = os.path.abspath(input)
        flag = cv2.IMREAD_COLOR
        image = cv2.imread(path, flag)
        delay = 3000
        return image, delay
    except AssertionError as e:
        logger.error(e)
        print(e)
        return None, None
    except Exception as e: 
        logger.error(e)
        print(e)
        return None, None
    """
    Publish worker status to MQTT topic.
    Pauses for rate second(s) between updates
    :return: None
    """
    while KEEP_RUNNING:
        payload = json.dumps({"Shopper": INFO.shopper, "Looker": INFO.looker, "Session": INFO.session, "totalSessions": INFO.totalSessions})
        time.sleep(1)
        CLIENT.publish(TOPIC, payload=payload)
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
def draw_results(frame, faces, det_time_fd, det_time_hp):
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
            "Shopper: {}".format(INFO.shopper),
            "Looker: {}".format(INFO.looker),
            "Sessions: {}".format(INFO.session)
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
    drawFaces(frame,faces,styles['boxes'])
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
def drawFaces(frame,faces, styles):
    # Draw boxes around faces
    for res_hp in faces:
        xmin, ymin, xmax, ymax = res_hp      
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), styles['color'], styles['thickness'])
    return frame
def cleanup(networks,vcap)
    # decommission networks
    for n in networks:
        n['network'].clean()
    if vcap:
        vcap.release()
    cv2.destroyAllWindows()
def processFrame(vcap,fd,hp,ret,next_frame,faces,args):
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
    looking = 0
    session = False
    if len(faces) > 0:
        # Look for poses
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
                looking += 1
                POSE_CHECKED = True
                # Check if looker is more than 75% of the camera h or w
                frame_wh = [vcap.get(3), vcap.get(4)]
                assert (frame_wh == initial_wh), "Error! frame_wh [{0}] != initial_wh [{1}]".format(frame_wh,initial_wh)
                # check to see if a new session should be started
                session = needNewSession(INFO.session,frame,face,frame_wh)
            else:
                pass
    else:
        det_time_hp = None
    updateSession(session)
    INFO = INFO._replace(looker=looking)

    #Update Total Sessions
    if session:
        # print("INFO.totalSessions: {0} {1}; INFO.sessions: {2} {3}; ".format(type(INFO.totalSessions),INFO.totalSessions,type(INFO.sessions),INFO.sessions))
        INFO = INFO._replace(totalSessions=(INFO.totalSessions + 1))
    # Draw performance stats
    draw_results(frame, faces, det_time_fd, det_time_hp)
    # Draw performance stats
    draw_results(frame, faces, det_time_fd, det_time_hp)

    return next_frame, faces, key_pressed
def getLookingFaces(image,faces,hp):
    if len(faces) > 0:
    # Look for head poses
    lookingFaces = []
    for face in faces:
        xmin, ymin, xmax, ymax = face
        # get head pose
        head_pose = image[ymin:ymax, xmin:xmax]
        image_hp = cv2.resize(head_pose, (hp['w'], hp['h']))
        image_hp = image_hp.transpose((2, 0, 1))
        image_hp = image_hp.reshape((hp['n'], hp['c'], hp['h'], hp['w']))
        inf_start_hp = time.time()
        hp['network'].exec_net(0, image_hp)
        hp['network'].wait(0)
        det_time_hp = time.time() - inf_start_hp

        # Parse head pose detection results
        angle_p_fc = hp['network'].get_output(0, "angle_p_fc")
        angle_y_fc = hp['network'].get_output(0, "angle_y_fc")
        if ((angle_y_fc > -22.5) & (angle_y_fc < 22.5) & (angle_p_fc > -22.5) &
                (angle_p_fc < 22.5)):
            lookingFaces.append(face)
        else:
            pass
    else:
        det_time_hp = None
    return lookingFaces, det_time_hp
def getFaces(image,fd):
    # Get intial width and height of video stream
    initial_wh = (image.shape[:2])
    image_fd = cv2.resize(image, (fd['w'], fd['h']))
    # Change data layout from HWC to CHW
    image_fd = image_fd.transpose((2, 0, 1))
    image_fd = image_fd.reshape((fd['n'], fd['c'], fd['h'], fd['w']))
    key_pressed = cv2.waitKey(int(delay))

    # Start asynchronous inference for specified request
    inf_start_fd = time.time()
    fd['network'].exec_net(0, image_fd)

    # Wait for the result
    fd['network'].wait(0)
    det_time_fd = time.time() - inf_start_fd

    # Results of the output layer of the network
    res_fd = fd['network'].get_output(0)

    # Parse face detection output
    faces = face_detection(res_fd, initial_wh)
    #TODO: Figure out how to write a true face detection function
    # existing face detection method takes in "res" which is the result of a model 
    # then parses it based on a conf interval.

    return faces, det_time_fd
def main():
    """
    Load the network and parse the output.
    :return: None
    """
    # # Setup and parse args
    # args = args_parser().parse_args()
    # # if video stream, Establish video stream
    # vcap, DELAY = initializeInput(args.input)
    # ret, frame = vcap.read()

    # testing inputs
    refFile = "mysteryface03.jpg"
    filePaths = [ "mysteryface04.jpg",
        "mysteryface05.jpg","mysteryface06.jpg",
        "mysteryface07.jpg","mysteryface08.jpg" ]
    refImage = initializeImageFileInput(refFile)
    refImages = [initializeImageFileInput(path) for path in filePaths]

    # Initialize the class
    fdmodel = {
        'name':'fd',
        'model':FACE_DETECTION_MODEL, 
        'device':args.device,
        'up1':1, 'up2':1, 'up3':0,
        'cpu_extension':args.cpu_extension,
        'plugin':None
    }
    fd = getModelOutput(**fdmodel) # {name,model,device,network,up1,up2,up3,cpu_extension,plugin,n,c,h,w,}
    hpmodel = {
        'name':'hp',
        'model':HEAD_POSITION_MODEL,
        'device':args.device,
        'up1':1, 'up2':3, 'up3':0,
        'cpu_extension':args.cpu_extension,
        'plugin':fd['plugin']
    } 
    hp = getModelOutput(**hpmodel) # {name,model,device,network,up1,up2,up3,cpu_extension,plugin,n,c,h,w,}
    rimodel = {
        'name':'ri',
        'model':FACE_REID_MODEL,
        'device':None,
        'up1':1, 'up2':3, 'up3':0,
        'cpu_extension':args.cpu_extension,
        'plugin':hp['plugin']
    } 
    ri = getModelOutput(**rimodel) # {name,model,device,network,up1,up2,up3,cpu_extension,plugin,n,c,h,w,}
    
    # Main loop.  Repeats for each frame
    for img in images:
        # parse image for faces and face positions
        faces, det_time_fd = getFaces(image,fd,ret,next_frame,faces,args)
        # Parse faces into those that are looking
        lookingFaces,timeToGetHp = getLookingFaces(faces,hp)
        # Draw performance stats
        draw_results(image, faces, det_time_fd, det_time_hp)
        # Draw performance stats
        draw_results(image, faces, det_time_fd, det_time_hp)

    #cleanup
    cleanup([fd,hp,ri], None)


if __name__ == '__main__':
    main()
    sys.exit()

