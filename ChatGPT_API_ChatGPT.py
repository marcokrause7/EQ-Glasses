# Step 1: Import the OpenAI package and set up your API key:
import openai

# Replace 'YOUR_API_KEY' with your actual API key
api_key = 'YOUR_API_KEY'
openai.api_key = api_key

# Step 2: Define a function to interact with the chatbot API:
def get_chatbot_response(prompt):
    response = openai.Completion.create(
        engine="text-davinci-002",  # You can choose a different engine if needed
        prompt=prompt,
        temperature=0.7,  # Controls the randomness of the response (0.0 to 1.0)
        max_tokens=150,   # Controls the maximum length of the response
    )
    return response.choices[0].text.strip()


# Step 3: Test the chatbot by providing a prompt and getting a response:
prompt = "What is the capital of France?"
response = get_chatbot_response(prompt)
print("Chatbot's Response:", response)
