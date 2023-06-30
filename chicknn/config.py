import os
import logging

## Configuration Variables
# Logging
console_noise = logging.INFO
file_noise = logging.INFO
logs_dir = "artifacts/logs/"
log_filepath = logs_dir + "logfile.log"
log_max_bytes = 1_000_000
log_backups = 2

# Camera Settings
images_dir = "artifacts/images/"
img_filepath = images_dir + "latest_frame.jpg"
rtsp_url = "rtsp://admin:noob@192.168.0.12:8554/Streaming/Channels/102"
collect_rtsp_img = True
save_rtsp_img = True
rtsp_retries = 5
rtsp_timeout_ms = 50
rtsp_buffer_size = 1

# ML Settings
agent_source = None #"OpenAI" # "Huggingface" 
hf_inference_endpoint = "https://api-inference.huggingface.co/models/bigcode/starcoder"
# OpenAssistant: "https://api-inference.huggingface.co/models/OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5"
hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN", "")
remote_tools = False
openai_model = "text-davinci-003"
openai_api_key = os.getenv("OPENAI_API_KEY", "")

predators = ["raccoon", "skunk", "cat", "dog", "animal", "human", "person", "man", "woman", "other"]
