# chicknn
* Fnd and prevent predators from attacking your chickens (or other pets)
* Uses an LLM with tools to identify potential predators, notify you, and trigger actions

## Installation
Currently installed and tested in Ubuntu WSL2.

`pip install -r requirements.txt`

## Usage
First set your Huggingface token and/or OpenAI API Key using:

`export HUGGING_FACE_HUB_TOKEN=<your_hf_token>`

`export OPENAI_API_KEY=<your_openai_api_key>`

Run using:

`python -m chicknn`
