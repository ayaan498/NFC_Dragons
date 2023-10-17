import openai

# Set your OpenAI API key here
api_key = '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'

# Initialize the OpenAI API client
openai.api_key = api_key

def chat_with_bot(prompt):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text

# Get the initial context from the user
initial_context = input("Enter the initial context: ")

# Initialize the conversation with the user-provided context
conversation = f"You: {initial_context}\nBot:"

# Main loop for chatting
print("Bot: Hello! How can I assist you today?")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break
    conversation += f"\nYou: {user_input}\nBot:"
    response = chat_with_bot(conversation)
    print(response.strip())
