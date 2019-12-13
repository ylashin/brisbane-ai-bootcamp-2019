import time
import datetime
import os
import sys
import asyncio
from six.moves import input
import threading
from azure.iot.device.aio import IoTHubModuleClient

# Imports for inferencing
import onnxruntime as rt
import numpy as np
import cv2

# Load YOLO labels
labels_file = open("labels.txt")
labels_string = labels_file.read()
labels = labels_string.split(",")
labels_file.close()
label_lookup = {}
for i, val in enumerate(labels):
    label_lookup[val] = i
print("YOLO labels have been loaded successfully")


# Loading ONNX model
print("Loading Tiny YOLO ONNX model ...")
start_time = time.time()
session = rt.InferenceSession('TinyYolo-onnx-v3.onnx')
print("YOLO ONNX model has been loaded successfully", time.time()-start_time,"s")


capture = cv2.VideoCapture("/dev/video0")

def resize_image(imageData):
    img = imageData.astype('float32')
    img = cv2.resize(img,(416,416))
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    return img

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def run_onnx(image):
    input_name = session.get_inputs()[0].name
    start_time = time.time()
    pred = session.run(None, {input_name: image})
    pred = np.array(pred[0][0])

    labels_file = open("labels.txt")
    labels = labels_file.read().split(",")

    objects_found = []

    post_start_time = time.time()
    
    tiny_yolo_cell_width = 13
    tiny_yolo_cell_height = 13
    num_boxes = 5
    tiny_yolo_classes = 20

    CONFIDENCE_THRESHOLD = 0.5

    # Goes through each of the 'cells' in tiny_yolo. Each cell is responsible for detecting 5 objects
    
    for by in range (0, tiny_yolo_cell_width):
        for bx in range (0, tiny_yolo_cell_height):
            for bound in range (0, num_boxes):

                channel = bound * 25
                tx = pred[channel][by][bx]
                ty = pred[channel+1][by][bx]
                tw = pred[channel+2][by][bx]
                th = pred[channel+3][by][bx]
                tc = pred[channel+4][by][bx]

                confidence = sigmoid(tc)

                p1 = channel + 5
                p2 = channel + 5 + tiny_yolo_classes
                class_out = pred[p1:p2,by,bx] # THE TRICK

                class_out = softmax(class_out)
                class_detected = np.argmax(class_out)
                display_confidence = class_out[class_detected] * confidence

                if display_confidence > CONFIDENCE_THRESHOLD:
                    object_found = (labels[class_detected], display_confidence)
                    objects_found.append(object_found)

    return  objects_found


def capture_and_predict():
    try:
        print("Capturing image ...")
        success,image = capture.read()
        image = resize_image(image)
        objects_found = run_onnx(image)

        print(datetime.datetime.now())

        if len(objects_found) == 0 :
            print("No objects found!")
        else:
            print(objects_found)

        print("=" * 40)

    except Exception as e:
        print('EXCEPTION:', str(e))

async def main():
    try:
        if not sys.version >= "3.5.3":
            raise Exception( "The sample requires python 3.5.3+. Current version of Python: %s" % sys.version )

        # The client object is used to interact with your Azure IoT hub.
        module_client = IoTHubModuleClient.create_from_edge_environment()

        # connect the client.
        await module_client.connect()

        # define behavior for halting the application
        def stdin_listener():
            while True:
                try:
                    capture_and_predict()

                    selection = input("Press Q to quit\n")
                    if selection == "Q" or selection == "q":
                        print("Quitting...")
                        break
                except:
                    time.sleep(2)

        print ( "Started the capturing module")

        # Run the stdin listener in the event loop
        loop = asyncio.get_event_loop()
        user_finished = loop.run_in_executor(None, stdin_listener)

        # Wait for user to indicate they are done listening for messages
        await user_finished

        # Finally, disconnect
        await module_client.disconnect()
        capture.release()

    except Exception as e:
        print ( "Unexpected error %s " % e )
        raise

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.close()

    # If using Python 3.7 or above, you can use following code instead:
    # asyncio.run(main())