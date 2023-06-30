import PIL
from PIL import Image
import transformers
import chicknn.config as config
import logging

logger = logging.getLogger(__name__)
agent = None

def initialize_agent():
    global agent
    if not agent:
        if config.agent_source == "Huggingface":
            agent = transformers.tools.HfAgent(config.hf_inference_endpoint, token=config.hf_token)
            logger.info("Huggingface Agent is initialized 💪")
        elif config.agent_source == "OpenAI":
            from transformers.tools import OpenAiAgent
            agent = OpenAiAgent(model=config.openai_model, api_key=config.openai_api_key)
            logger.info("OpenAI Agent is initialized 💪")
        else:
            logger.warning(f"Agent ({config.agent_source}) not initialized")

def generate_image(prompt: str="dog standing next to a chicken"):
    im = agent.run(f"Generate an image with prompt: {prompt}")
    return im

def caption_image(image):
    try:
        if config.agent_source:
            caption = agent.run("Caption `img`",
                                img = image,
                                remote=config.remote_tools)
        else:
            image_captioner = transformers.load_tool('image-captioning',
                                                    remote=config.remote_tools)
            caption = image_captioner(image)
    except Exception as e:
        logger.error(e)
        caption = None
  
    return caption

def classify_text(text:str, labels:list):
    try:
        if config.agent_source:
            classification = agent.run("Classify `text` as `labels`",
                                        text=text,
                                        labels=labels,
                                        remote=config.remote_tools)
        else:
            text_classifier = transformers.load_tool("text-classification",
                                                    remote=config.remote_tools)
            classification = text_classifier(text, labels=labels)
    except Exception as e:
        logger.error(e)
        classification = None
    
    return classification

def question_text(text:str, labels:list):
    try:
        if config.agent_source:
            classification = agent.run("Classify `text` as `labels`",
                                        text=text,
                                        labels=labels,
                                        remote=config.remote_tools)
        else:
            text_qa = transformers.load_tool("text-question-answering",
                                                    remote=config.remote_tools)
            answer = text_qa(text, f"Does this text contain any of {labels}?")
    except Exception as e:
        logger.error(e)
        answer = None
    
    return answer

def question_image(image, labels:list):
    try:
        if config.agent_source:
            answer = agent.run("image_qa for `img`: does this image contain any of `labels`?",
                                img=image,
                                label=labels,
                                remote=config.remote_tools)
        else:
            image_qa = transformers.load_tool("image-question-answering",
                                                    remote=config.remote_tools)
            answer = image_qa(image, f"Does this image contain any of {labels}?")
    except Exception as e:
        logger.error(e)
        answer = None
    
    return answer

def verify_image_segmentation(image, label:str):
    try:
        if config.agent_source:
            mask = agent.run("Segmentation mask classification of `img` with label `label`", 
                        img=image.resize((1028,1028)), 
                        label=label,
                        remote=config.remote_tools)
        else:
            segmenter = transformers.load_tool("image-segmentation",
                                                remote=config.remote_tools)
            mask = segmenter(image, label=label)

        if mask:
            mask_classification = True
        else:
            mask_classification = False
    except Exception as e:
        logger.error(e)
        mask = None
        mask_classification = None
    
    return mask_classification

def analyze_image(image, labels:list):
    caption = caption_image(image)
    classification = classify_text(caption, labels)
    question = question_image(image, classification)
    verify_segmentation = verify_image_segmentation(image, classification)

    img_results = {
        "caption": caption,
        "classification": classification,
        "question_image": question,
        "verify_image_segmentation": verify_segmentation 
    }

    return img_results


