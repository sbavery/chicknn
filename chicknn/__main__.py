# main.py
import logging
from PIL import Image
from datetime import datetime
import multiprocessing
from chicknn.util import setup_logger
import chicknn.config as config
import chicknn.camera as camera
import chicknn.agents as agents

logger = None

def main():
    global logger
    logger = setup_logger(console_level=config.console_noise,
                          file_level=config.file_noise,
                          filename=config.log_filepath)
    while True:
        try:
            if config.collect_rtsp_img:
                # Get latest camera frame
                logger.debug(f"Using RTSP URL {config.rtsp_url}")
                logger.debug("Getting latest camera frame")
                image = camera.get_latest_frame(config.rtsp_url)
                
                if config.save_rtsp_img:
                    image.save(config.img_filepath)
                    logger.debug(f"Image saved to {config.img_filepath}")
            else:
                logger.debug(f"Collecting image from {config.img_filepath}")
                image = Image.open(config.img_filepath)

            # Save and analyze the image if it's available
            if image:
                if config.agent_source: agents.initialize_agent()
                caption = agents.caption_image(image)
                
                # img_results = agents.analyze_image(image, config.predators)
                # for result in img_results.keys():
                #     logger.info(f"{result}: {img_results[result]}")
                # caption = img_results["caption"]

                logger.info(caption)

                for label in config.predators:
                    if label in caption:
                        logger.warning(f"Found {label}")
                        datestring = datetime.today().strftime('%Y-%m-%dT%H-%M-%S.%f')[:-3]
                        img_name = f"{datestring}_{label}.jpg"
                        image.save(f"{config.images_dir}{img_name}")
                        logger.info(f"Saved image {img_name}")

            else:
                logger.error("No image found")
        except Exception as e:
            logger.error(e)

if __name__ == '__main__':
    main()