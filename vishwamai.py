# Ultra Clean MCP Distillation Framework 🚀🔥

import os
# Force JAX to use CPU
os.environ['JAX_PLATFORMS'] = 'cpu'

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
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.output_dim)(x)
        return x

    def get_initial_params(self, rng, input_shape):
        """Helper function to get initial parameters."""
        variables = self.init(rng, jnp.ones(input_shape))
        return variables['params']

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

# --- Loss and Training Functions ---
@jax.jit
def compute_loss(params, state, batch):
    """Compute the MSE loss."""
    inputs, targets = batch
    predictions = state.apply_fn({'params': params}, inputs)
    return jnp.mean((predictions - targets) ** 2)

@jax.jit
def train_step(state, batch):
    """Perform a single training step."""
    loss_value, grads = jax.value_and_grad(compute_loss)(
        state.params, state, batch
    )
    state = state.apply_gradients(grads=grads)
    return state, loss_value

def collect_teacher_responses(num_samples=50, input_dim=16):
    """Collect responses from the teacher model"""
    try:
        x_data = np.random.randn(num_samples, input_dim).astype(np.float32)
        teacher_vectors = []

        print("Querying Ollama Teacher...")
        for i in range(num_samples):
            try:
                prompt = f"Distill this input: {x_data[i].tolist()}"
                teacher_text = get_teacher_response(prompt)
                print(f"[{i+1}/{num_samples}] Teacher response: {teacher_text[:50]}...")
                vector = text_to_vector(teacher_text)
                teacher_vectors.append(vector)
            except Exception as e:
                print(f"Error processing sample {i}: {str(e)}")
                continue

        if not teacher_vectors:
            raise ValueError("No valid teacher responses were collected")

        teacher_vectors = np.array(teacher_vectors, dtype=np.float32)
        print(f"Successfully collected {len(teacher_vectors)} teacher responses")
        return x_data, teacher_vectors
    except Exception as e:
        print(f"Fatal error in data collection: {str(e)}")
        raise

def main():
    try:
        # Collect teacher responses
        x_data, teacher_vectors = collect_teacher_responses()

        # Initialize model and training state
        model = StudentModel()
        rng = jax.random.PRNGKey(0)
        
        # Get initial parameters
        input_shape = (1, 16)  # batch_size=1, input_dim=16
        params = model.get_initial_params(rng, input_shape)

        # Create optimizer
        learning_rate = 1e-3
        tx = optax.adam(learning_rate)

        # Create training state
        state = train_state.TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=tx,
        )

        # Training loop
        print("\nStarting MCP distillation training 🚀...")
        num_samples = len(x_data)
        
        # Convert data to JAX arrays
        x_data = jnp.array(x_data)
        teacher_vectors = jnp.array(teacher_vectors)

        total_loss = 0.0
        losses = []
        
        for i in range(num_samples):
            try:
                # Prepare batch
                inputs = x_data[i:i+1]
                targets = teacher_vectors[i:i+1]
                batch = (inputs, targets)

                # Training step
                state, loss = train_step(state, batch)
                losses.append(loss)

                # Report progress
                if (i + 1) % 10 == 0:
                    avg_loss = jnp.mean(jnp.array(losses[-10:]))
                    print(f"Progress: {i+1}/{num_samples} samples processed (avg_loss: {avg_loss:.4f})")

            except Exception as e:
                print(f"Error during training step {i}: {str(e)}")
                continue

        print("✅ MCP distillation (1 epoch) complete!")
        
    except Exception as e:
        print(f"Fatal error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()
