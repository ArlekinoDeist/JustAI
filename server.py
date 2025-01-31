from flask import Flask, request, jsonify
from transformers import LlamaForCausalLM, LlamaTokenizer

app = Flask(__name__)

# Загрузка модели и токенизатора
model = LlamaForCausalLM.from_pretrained("path_to_llama_model")
tokenizer = LlamaTokenizer.from_pretrained("path_to_llama_model")

@app.route('/ask', methods=['POST'])
def ask():
    # Получаем входные данные
    user_input = request.json.get('input')
    
    # Проверка на пустой ввод
    if not user_input:
        return jsonify({"error": "No input provided"}), 400
    
    # Токенизация
    inputs = tokenizer(user_input, return_tensors="pt")
    
    # Генерация ответа
    outputs = model.generate(**inputs, max_length=100, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
