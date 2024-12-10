import tkinter as tk
import chad_gpt as chad
import torch

# Main GUI window setup
window = tk.Tk()
window.title("Chadbot")
window.geometry("500x400")

# Placeholder text for the entry box
placeholder_text = "Ask a question"

# Textbox for the bot's responses
output = tk.Text(window, wrap="word", height=15)
output.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=10, pady=(10, 0))

# Generates a response to the user's input
def generate_response(user_input):
    context = torch.zeros((1, 1), dtype=torch.long)
    return "Bot: " + chad.decode(chad.model.generate(context, max_gens=500)[0].tolist())

# Type animation that will type out the text character by character for chatbot's response
def type_out_text(text, widget, idx=0):
    if idx < len(text):
        widget.insert(tk.END, text[idx])
        widget.update()
        window.after(50, type_out_text, text, widget, idx + 1)

# Responds to the user's input, and adds the user's input and the bot's response to the output textbox with animated typing response,
# and then after question is asked it will automatically clear the entry box and add the placeholder text
def respond():
    user_input = entry.get()
    if user_input.strip() == "" or user_input == placeholder_text:
        return
    output.insert(tk.END, f"You: {user_input}\n")
    entry.delete(0, tk.END)
    window.focus()
    add_placeholder()
    response = generate_response(user_input)
    type_out_text(response + "\n", output)
 
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
 
# Entry box with placeholder
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
