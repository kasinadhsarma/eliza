# Ultra Clean MCP Distillation Framework 🚀🔥

import requests
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state
import numpy as np
from sentence_transformers import SentenceTransformer

# --- Ollama Teacher Setup ---
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:latest"

# --- Student Model ---
class StudentModel(nn.Module):
    hidden_dim: int = 128
    output_dim: int = 384

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.output_dim)(x)
        return x

# --- Call Ollama (Teacher) ---
def get_teacher_response(prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(OLLAMA_URL, json=payload)
    return response.json()["response"]

# --- Encode Text to Vector ---
encoder = SentenceTransformer('all-MiniLM-L6-v2')  # Small, fast sentence encoder

def text_to_vector(text: str) -> np.ndarray:
    return encoder.encode(text)

# --- Loss Function ---
def distillation_loss(params, inputs, targets, model):
    preds = model.apply({'params': params}, inputs)
    return jnp.mean(optax.l2_loss(preds, targets))

# --- Train Step ---
@jax.jit
def train_step(state, batch_inputs, batch_targets):
    grads = jax.grad(distillation_loss)(state.params, batch_inputs, batch_targets, state.apply_fn)
    return state.apply_gradients(grads=grads)

# --- Simulate Dataset ---
x_data = np.random.randn(50, 16).astype(np.float32)  # 50 fake inputs
teacher_vectors = []

print("Querying Ollama Teacher...")
for i in range(len(x_data)):
    prompt = f"Distill this input: {x_data[i].tolist()}"
    teacher_text = get_teacher_response(prompt)
    print(f"[{i}] Teacher response: {teacher_text[:50]}...")
    vector = text_to_vector(teacher_text)
    teacher_vectors.append(vector)

teacher_vectors = np.array(teacher_vectors, dtype=np.float32)

# --- Student Initialization ---
student = StudentModel()
key = jax.random.PRNGKey(0)
params = student.init(key, jnp.ones((1, 16)))['params']

tx = optax.adamw(learning_rate=1e-3)
state = train_state.TrainState.create(apply_fn=student.apply, params=params, tx=tx)

# --- Training (MCP Mode: 1 Epoch) ---
print("Starting MCP distillation training 🚀...")
for i in range(len(x_data)):
    inputs = x_data[i:i+1]
    targets = teacher_vectors[i:i+1]
    state = train_step(state, inputs, targets)

print("✅ MCP distillation (1 epoch) complete!")
