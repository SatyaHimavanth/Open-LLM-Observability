import dspy
import universal_agent_obs
import time

def main():
    # Configure DSPy with Ollama (assuming local instance, but it will work with the interceptor anyway)
    # The interceptor catches calls to dspy.LM regardless of the provider
    lm = dspy.LM('ollama_chat/qwen3.5:2b', api_base='http://localhost:11434', api_key='none')
    dspy.settings.configure(lm=lm)

    # Use a signature
    class BasicQA(dspy.Signature):
        """Answer questions with short factoid answers."""
        question = dspy.InputField()
        answer = dspy.OutputField(desc="often between 1 and 5 words")

    # Define a simple module
    generate_answer = dspy.Predict(BasicQA)

    # Attach trace context via a small callback handler instead of global set_context
    # This lets the interceptor pick up per-run user/tags/metadata.
    # Project name is resolved automatically from AGENT_OBS_PROJECT env var.
    try:
        from universal_agent_obs.dspy import TraceContextCallbackHandler
        trace_cb = TraceContextCallbackHandler(
            user={"id": "demo-user", "name": "DSPy Tester"},
            tags=["sample", "dspy"],
        )
        # attach to the LM instance so the interceptor can find it
        setattr(lm, "_obs_callbacks", [trace_cb])
    except Exception:
        # fall back to global set_context if helper not available
        universal_agent_obs.set_context(
            user={"id": "demo-user", "name": "DSPy Tester"},
            tags=["sample", "dspy"]
        )

    print("Running DSPy prediction...")
    try:
        pred = generate_answer(question="What is the capital of France?")
        print(f"Question: What is the capital of France?")
        print(f"Predicted Answer: {pred.answer}")
    except Exception as e:
        print(f"Prediction failed (expected if Ollama is not running): {e}")
    finally:
        universal_agent_obs.flush(timeout=5)