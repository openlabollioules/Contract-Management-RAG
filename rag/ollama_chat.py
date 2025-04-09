import subprocess

def ask_ollama(prompt, model="llama3"):
    process = subprocess.Popen(["ollama", "run", model],
                               stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE)
    output, _ = process.communicate(input=prompt.encode())
    return output.decode()
