from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda:0"  # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(
    "qwen2-0.5b-finetuned/checkpoint-432", torch_dtype="auto", device_map=device
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")

false, true = False, True
datapoint = {
    "state": {
        "turn": false,
        "width": 10,
        "height": 10,
        "units": [
            {
                "id": {"id": "0G"},
                "position": {"x": 3, "y": 7},
                "health": 5,
                "max_health": 5,
                "attack": 1,
                "hit_rate": 0.5,
                "movement_range": 0,
                "max_movement_range": 1,
                "has_attacked": false,
                "player_id": false,
            },
            {
                "id": {"id": "0Z"},
                "position": {"x": 4, "y": 3},
                "health": 5,
                "max_health": 5,
                "attack": 1,
                "hit_rate": 0.5,
                "movement_range": 0,
                "max_movement_range": 1,
                "has_attacked": true,
                "player_id": false,
            },
            {
                "id": {"id": "0I"},
                "position": {"x": 5, "y": 4},
                "health": 5,
                "max_health": 5,
                "attack": 1,
                "hit_rate": 0.5,
                "movement_range": 0,
                "max_movement_range": 1,
                "has_attacked": false,
                "player_id": false,
            },
            {
                "id": {"id": "1U"},
                "position": {"x": 7, "y": 8},
                "health": 5,
                "max_health": 5,
                "attack": 1,
                "hit_rate": 0.5,
                "movement_range": 1,
                "max_movement_range": 1,
                "has_attacked": false,
                "player_id": true,
            },
            {
                "id": {"id": "1L"},
                "position": {"x": 4, "y": 2},
                "health": 4,
                "max_health": 5,
                "attack": 1,
                "hit_rate": 0.5,
                "movement_range": 1,
                "max_movement_range": 1,
                "has_attacked": false,
                "player_id": true,
            },
            {
                "id": {"id": "1H"},
                "position": {"x": 1, "y": 0},
                "health": 5,
                "max_health": 5,
                "attack": 1,
                "hit_rate": 0.5,
                "movement_range": 1,
                "max_movement_range": 1,
                "has_attacked": false,
                "player_id": true,
            },
        ],
    },
    "action": {"action_type": "end_turn", "player_id": false},
}

example_prompt = repr(datapoint.get("state", "")).strip()

messages = [
    {"role": "user", "content": example_prompt},
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=100)
generated_ids = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
print(len(response))
