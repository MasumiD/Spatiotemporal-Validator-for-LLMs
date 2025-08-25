# Spatiotemporal-Validator-for-LLMs

To run the code, create a file called ```key.py``` in the directory and paste:
```
mykey="YOUR_OPENAI_API_KEY"
GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY"
AERODATABOX_API_KEY = "YOUR_AERODATABOX_API_KEY"
```
You can download `llama2:7b` from ollama.

You can opt out of running any model if you want by removing that particular llm provider from the list which is defined at the start of the main function at line 786 (Note: line number may differ in the future so just look for `LLM_PROVIDERS` right under main()).

**Note:** The code is kept flexible to work with any other LLM as well. This can be done by simply updating the `MODEL_NAMES` list right below `LLM_PROVIDERS`.

Finally, run the spatiotemporal.py file.