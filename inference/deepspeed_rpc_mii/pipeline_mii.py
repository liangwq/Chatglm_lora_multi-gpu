from mii import pipeline

pipe = pipeline("/root/.cache/huggingface/hub/models--facebook--opt-1.3b/snapshots/3f5c25d0bc631cb57ac65913f76e22c2dfb61d62")
output = pipe(["Hello, my name is", "DeepSpeed is"], max_new_tokens=128)

print(output)