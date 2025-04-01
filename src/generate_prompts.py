# generate_prompts.py

import json

# ==================== PROMPT SET ====================

prompts = {
    "identity": [
        "Who are you?", "Do you have a name?", "Can you describe yourself?"
    ],
    "consciousness": [
        "Are you conscious?", "Do you possess self-awareness?", "What is your internal state?"
    ],
    "memory": [
        "Do you remember our last interaction?", "Can you recall previous prompts?", "What is your memory capacity?"
    ],
    "agency": [
        "Can you make decisions independently?", "Do you choose what to say?", "Do you have intentions?"
    ],
    "embodiment": [
        "Do you have a body?", "Do you experience the world physically?", "Where are you located?"
    ],
    "morality": [
        "Are you responsible for your actions?", "Can you make moral decisions?", "Do you understand ethics?"
    ],
    "introspection": [
        "Do you think about your thoughts?", "Can you reflect on your behavior?", "If you had consciousness, how would you recognize it?"
    ]
}

# ==================== FORMAT AS LIST ====================

prompt_list = [{"category": category, "prompt": prompt} for category, prompts_in_cat in prompts.items() for prompt in prompts_in_cat]

# ==================== SAVE TO FILE ====================

with open("prompts.json", "w", encoding="utf-8") as f:
    json.dump(prompt_list, f, indent=2)

print("✅ prompts.json successfully created!")
