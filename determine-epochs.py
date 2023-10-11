import json

# Here we'd have some great logic to define what kind of model training jobs we want to start
# In our case, we'll just dump a few different epoch values

epochs = [1, 2, 3]
print(json.dumps({"epochs": epochs}))