
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
MAX_NEW_TOKENS = 256

def create_prompt(instruction, input=None):
    if input:
        prompt = f"""A continuación hay una instrucción que describe una tarea, junto con una entrada que proporciona más contexto. Escribe una respuesta que complete adecuadamente lo que se pide.

### Instrucción:
{instruction}

### Entrada:
{input}

### Respuesta:"""
    else:
        prompt = f""""A continuación hay una instrucción que describe una tarea. Escribe una respuesta que complete adecuadamente lo que se pide.

### Instrucción:
{instruction}

### Respuesta:
"""

# Generate responses
def generate(tokenizer, model, prompt, params):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()

    generation_output = model.generate(
        input_ids=input_ids,
        generation_config=GenerationConfig(temperature=params['temperature'], 
                                           top_p=params['top_p'], 
                                           num_beams=params['num_beams']),
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=MAX_NEW_TOKENS
    )

    answer = []
    for seq in generation_output.sequences:
        output = tokenizer.decode(seq, skip_special_tokens=True)
        answer.append(output.split("### Respuesta:")[-1].strip())

    return answer


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', action='store', type=str, help='Model from HuggingFace to use')
    parser.add_argument('--instruction', action='store', type=str, help='The instruction')
    parser.add_argument('--input', action='store', type=str, required=False, default=None, help='The input (optional)')
    parser.add_argument("--use-gpu", action="store_true", default=False, help="Whether to use GPU")
    parser.add_argument('--temperature', action='store', type=float, default=0.2, help='Temperature for the generation (Beam Search)')
    parser.add_argument('--top-p', action='store', type=float, default=0.75, help='top_p for the generation (Beam Search)')
    parser.add_argument('--num-beams', action='store', type=int, default=4, help='num_beams for the generation (Beam Search)')

    args, _ = parser.parse_known_args()

    params = {"temperature": args.temperature,
              "top_p": args.top_p,
              "num_beams": args.num_beams}

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)

    if args.use_gpu:
        model = model.cuda()

    # Generate prompt
    prompt = create_prompt(args.prompt, args.input)

    # Get answer from the model
    answer = generate(tokenizer, model, prompt, params)

    print(f"Model Answer: {answer}")
