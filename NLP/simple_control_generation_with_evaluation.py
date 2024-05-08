
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer

nltk.download('punkt')

# Initialize models and tokenizers
tiny_lama_model = AutoModelForSeq2SeqLM.from_pretrained("tiny-lama-checkpoint")
tiny_lama_tokenizer = AutoTokenizer.from_pretrained("tiny-lama-checkpoint")

flan_t5_model = T5ForConditionalGeneration.from_pretrained("flan-t5-checkpoint")
flan_t5_tokenizer = T5Tokenizer.from_pretrained("flan-t5-checkpoint")

def generate_control(model, tokenizer, process_description, risk_description, model_name="tiny-lama"):
    input_text = f"Process: {process_description}, Risk: {risk_description}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(input_ids)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def evaluate(predictions, references):
    # Tokenize sentences for BLEU score
    predictions_tokens = [nltk.word_tokenize(pred.lower()) for pred in predictions]
    references_tokens = [[nltk.word_tokenize(ref.lower())] for ref in references]
    bleu_scores = [sentence_bleu(ref, pred) for pred, ref in zip(predictions_tokens, references_tokens)]

    # Calculate METEOR score
    meteor_scores = [meteor_score([ref], pred) for pred, ref in zip(predictions, references)]

    # Calculate ROUGE scores
    rouge = Rouge()
    rouge_scores = rouge.get_scores(predictions, references, avg=True)

    return {
        "BLEU": sum(bleu_scores) / len(bleu_scores),
        "METEOR": sum(meteor_scores) / len(meteor_scores),
        "ROUGE-L": rouge_scores['rouge-l']['f']  # Focus on ROUGE-L F1-score for sentence level similarity
    }

# Example usage
if __name__ == "__main__":
    new_process = "Software development involving multiple teams."
    new_risk = "Risk of code inconsistencies leading to bugs."
    control = generate_control(tiny_lama_model, tiny_lama_tokenizer, new_process, new_risk, "tiny-lama")
    print("Control Suggestion (tiny-lama):", control)
    
    # Example evaluations
    predicted_controls = [control]
    reference_controls = ["Implement strict code review processes with automated testing."]
    results = evaluate(predicted_controls, reference_controls)
    print("Evaluation Results:", results)
