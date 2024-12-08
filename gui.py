import tkinter as tk
 
# Main GUI window setup
window = tk.Tk()
window.title("Chadbot")
window.geometry("400x300")
 
# Placeholder text for the entry box
placeholder_text = "Ask a question"
 
# Textbox for the bot's responses
output = tk.Text(window, wrap="word", height=15)
output.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=10, pady=(10, 0))
 
# Generates a response to the user's input
def generate_response(user_input):
    return "Bot: " + user_input[::-1] # We'll just reverse the input for now, and then put the variable name or method name to execute the response of the bot
 
# Responds to the user's input, and adds the user's input and the bot's response to the output textbox,
# and then after question is asked it will automatically clear the entry box and add the placeholder text
def respond():
    user_input = entry.get()
    if user_input.strip() == "" or user_input == placeholder_text:
        return
    response = generate_response(user_input)
    output.insert(tk.END, f"You: {user_input}\n{response}\n")
    entry.delete(0, tk.END)
    window.focus()
    add_placeholder()
 
# Adds the placeholder "Ask a question" text when the entry box is empty
def add_placeholder():
    if not entry.get():
        entry.insert(0, placeholder_text)
        entry.config(fg="grey")
 
# Removes the placeholder "Ask a question" text when the entry box is clicked
def remove_placeholder(event=None):
    if entry.get() == placeholder_text:
        entry.config(fg="white")
        entry.delete(0, tk.END)
 
# Entry box with placeholder, and removing placeholder when clicked
entry = tk.Entry(window, fg="grey")
entry.grid(row=1, column=0, sticky="ew", padx=(10, 5), pady=10)
entry.insert(0, placeholder_text)
entry.bind("<FocusOut>", lambda event: add_placeholder())
entry.bind("<FocusIn>", remove_placeholder)
 
# "Ask" button
submit = tk.Button(window, text="Ask", command=respond)
submit.grid(row=1, column=1, sticky="ew", padx=(5, 10), pady=10)
 
# Grid layout for window resizing
window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)
 
# Start the GUI
window.mainloop()