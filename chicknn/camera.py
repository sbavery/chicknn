import cv2
from PIL import Image
from retrying import retry
import chicknn.config as config
import logging

logger = logging.getLogger(__name__)
cap = None

def set_buffer_size(cap, buffer_size):
    # Check if the buffer size property is supported
    if not cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size):
        logger.warning("Unable to set RTSP buffer size")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

def clear_buffer(cap):
    # Read frames until the most recent frame is reached
    while True:
        ret, frame = cap.read()
        if not ret:
            break

@retry(stop_max_attempt_number=config.rtsp_retries, wait_fixed=config.rtsp_timeout_ms)
def read_frame(cap):
    # Read the next frame from the stream
    ret, frame = cap.read()

    # Check if a frame is successfully captured
    if not ret:
        logger.error("Failed to capture frame from the RTSP stream.")
        raise Exception("Failed to capture frame")

    return ret, frame

def get_latest_frame(rtsp_url):
    logger.debug("Open RTSP stream")
    cap = cv2.VideoCapture(rtsp_url)
    # set_buffer_size(cap, config.rtsp_buffer_size)
    
    # Check if the stream is opened correctly
    if not cap.isOpened():
        logger.error("Unable to open RTSP stream")
        return None
    
    try:
        # Capture the frame within the specified timeout
        ret, frame = read_frame(cap)
    except Exception:
        logger.warning("Failed to capture frame within the timeout")
        return None
    
    # Release the stream
    logger.debug("Release the stream")
    cap.release()
    
    # Convert the OpenCV frame to a Pillow image
    if ret:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return image
    
    return None