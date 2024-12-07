import tkinter as tk

def generate_response(user_input):
    # Placeholder function for generating a response
    return "Bot: " + user_input[::-1] # We'll just reverse the input for now, and then put the variable name or method name to execute the response of the bot

def respond():
    user_input = entry.get()
    if user_input.strip() == "" or user_input == placeholder_text:
        return  # Ignore empty input or placeholder text
    response = generate_response(user_input)
    output.insert(tk.END, f"You: {user_input}\n{response}\n")
    entry.delete(0, tk.END)  # Clear the entry box
    window.focus()  # Shift focus to the main window to enable placeholder functionality
    add_placeholder()  # Re-add placeholder immediately

def add_placeholder():
    """Adds the placeholder text if the entry is empty."""
    if not entry.get():  # Check if the entry is empty
        entry.insert(0, placeholder_text)
        entry.config(fg="grey")

def remove_placeholder(event=None):
    """Removes the placeholder text when the user focuses on the entry."""
    if entry.get() == placeholder_text:
        entry.delete(0, tk.END)
        entry.config(fg="white")  # Switch text color for actual input

# Main window setup
window = tk.Tk()
window.title("Chadbot")
window.geometry("400x300")

placeholder_text = "Ask a question"

# Output textbox (aligned at the top)
output = tk.Text(window, wrap="word", height=15)
output.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=10, pady=(10, 0))

# Entry box with placeholder
entry = tk.Entry(window, fg="grey")
entry.grid(row=1, column=0, sticky="ew", padx=(10, 5), pady=10)
entry.insert(0, placeholder_text)  # Add placeholder initially
entry.bind("<FocusIn>", remove_placeholder)
entry.bind("<FocusOut>", lambda event: add_placeholder())

# Submit button
submit = tk.Button(window, text="Ask", command=respond)
submit.grid(row=1, column=1, sticky="ew", padx=(5, 10), pady=10)

# Configure grid layout
window.grid_rowconfigure(0, weight=1)  # Let the textbox expand vertically
window.grid_columnconfigure(0, weight=1)  # Let the entry box expand horizontally

window.mainloop()
