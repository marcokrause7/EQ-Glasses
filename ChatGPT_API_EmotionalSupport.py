# Resources: AnalyticsVidhya and ChatGPT
# Step 1: Import the OpenAI package and set up your API key in an Environment Variable for security:
import openai
import os
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

# Access the API key from the environment variable
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key

#Define a function that can be used to get a response from ChatGPT
def get_completion(prompt, model="gpt-3.5-turbo"):

    messages = [{"role": "user", "content": prompt}]

    response = openai.ChatCompletion.create(

    model=model,

    messages=messages,

    temperature=0,

    )

    return response.choices[0].message["content"]


# Query the API

# The prompt will be connected with PyTorch in order to input the necessary change for the emotional status of target
'''
from PytorchCopy import emotion

prompt = "The person in front of me is {emotion}, what can I do to help?".format(emotion)

response = get_completion(prompt)

print(response)
'''