import pandas as pd
from datasets import Dataset
import re
import uuid

# Function to detect Arabic text
def contains_arabic(text):
    if isinstance(text, str):
        arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+')
        return bool(arabic_pattern.search(text))
    return False

# Function to clean text data
def clean_text(text):
    if isinstance(text, str):
        # Remove extra whitespace and quotes, preserve special characters
        cleaned = text.strip().replace('^"|"$', '')
        # Normalize Unicode for French and Arabic
        cleaned = cleaned.encode('utf-8').decode('utf-8')
        return cleaned
    return text

# Function to preprocess the dataset
def preprocess_dataset(csv_file):
    try:
        # Load CSV file with UTF-8 encoding
        df = pd.read_csv(csv_file, encoding='utf-8')
        # Check if CSV has at least two columns
        if len(df.columns) < 2:
            raise ValueError("CSV file must have at least two columns (questions and answers).")

        # Rename first two columns to 'question' and 'answer'
        df = df.iloc[:, [0, 1]].copy()  # Select first two columns
        df.columns = ['question', 'answer']  # Rename them

        # Clean questions and answers
        df['question'] = df['question'].apply(clean_text)
        df['answer'] = df['answer'].apply(clean_text)

        # Remove empty or invalid rows
        df = df.dropna(subset=['question', 'answer'])
        df = df[df['question'].str.strip() != '']
        df = df[df['answer'].str.strip() != '']

        # Check for Arabic content
        arabic_questions = df['question'].apply(contains_arabic).sum()
        arabic_answers = df['answer'].apply(contains_arabic).sum()
        print(f"Arabic text detected: {arabic_questions} questions, {arabic_answers} answers")

        # Convert to Hugging Face Dataset
        dataset = Dataset.from_pandas(df[['question', 'answer']])

        return dataset

    except Exception as e:
        print(f"Error processing CSV file: {str(e)}")
        raise

# Main execution
def main():
    # File path to the CSV (updated for Colab)
    csv_file = "/content/File.csv"

    # Preprocess the dataset
    dataset = preprocess_dataset(csv_file)

    # Split into train and validation sets (80-20 split)
    train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = train_test_split['train']
    val_dataset = train_test_split['test']

    # Save the datasets (updated for Colab)
    train_dataset.save_to_disk("/content/train_chatbot_dataset")
    val_dataset.save_to_disk("/content/val_chatbot_dataset")

    print("Dataset preprocessing complete. Train and validation datasets saved.")

if __name__ == "__main__":
    main()