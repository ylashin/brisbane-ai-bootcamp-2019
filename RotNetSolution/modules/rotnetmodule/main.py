import time
import os
import sys
import asyncio
from six.moves import input
import threading
from azure.iot.device.aio import IoTHubModuleClient

import requests

# Imports for inferencing
import onnxruntime as rt
import numpy as np
import cv2

# Loading ONNX model
print("loading RotNet ...")
start_time = time.time()

session = rt.InferenceSession('rotnet_street_view_mobilenet_v2.onnx')
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name 

print("loaded after", time.time()-start_time,"s")

def preprocess_input(x):
    x = x[..., ::-1]
    mean = [103.939, 116.779, 123.68]
    x[..., 0] -= mean[0]
    x[..., 1] -= mean[1]
    x[..., 2] -= mean[2]

    return x

def predict_rotation():
    try:
        print("Downloading input image ...")
        web_response = requests.get('https://bigdisk.blob.core.windows.net/public/image.png')
        raw_contents = web_response.content

        image = np.asarray(bytearray(raw_contents), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = preprocess_input(image.astype(np.float32))
        image = np.expand_dims(image, axis=0)
        print("Running RotNet prediction ...")

        prediction = session.run([output_name], {input_name: image})
        angle = np.argmax(prediction[0][0])
        print("Rotation angle is: " + str(angle))
        print("=" * 40)
    except Exception as error:
        print("Unexpected error %s " % error )

async def main():
    try:
        if not sys.version >= "3.5.3":
            raise Exception( "The sample requires python 3.5.3+. Current version of Python: %s" % sys.version )
        print ( "IoT Hub Client for Python" )

        # The client object is used to interact with your Azure IoT hub.
        module_client = IoTHubModuleClient.create_from_edge_environment()

        # connect the client.
        await module_client.connect()

        # define behavior for halting the application
        def stdin_listener():
            while True:
                try:
                    predict_rotation()
                    selection = input("Press Q to quit\n")
                    if selection == "Q" or selection == "q":
                        print("Quitting...")
                        break
                except:
                    time.sleep(5)

        # Run the stdin listener in the event loop
        loop = asyncio.get_event_loop()
        user_finished = loop.run_in_executor(None, stdin_listener)

        # Wait for user to indicate they are done listening for messages
        await user_finished

        # Finally, disconnect
        await module_client.disconnect()

    except Exception as e:
        print ( "Unexpected error %s " % e )
        raise

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.close()

    # If using Python 3.7 or above, you can use following code instead:
    # asyncio.run(main())