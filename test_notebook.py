from reports.notebook_generator import generate_notebook_bytes
import json

dummy_history = [
    {"role": "user", "content": "Show me a chart"},
    {"role": "assistant", "content": "Here it is:", "metadata": {"code": "import matplotlib.pyplot as plt\nplt.plot([1, 2, 3])"}}
]

nb_bytes = generate_notebook_bytes(dummy_history)
with open("test_output.ipynb", "wb") as f:
    f.write(nb_bytes)
print("Notebook generated successfully. Size:", len(nb_bytes))
