# Comparative Analysis of Fine-Tuning vs. Multishot Prompting Techniques for Summarizing Albanian Parliamentary Speeches

## Introduction

This project evaluates two methodologies in Natural Language Processing (NLP) to summarize Albanian parliamentary speeches: fine-tuning a pre-trained Language Model (LM) and multishot prompt engineering. The aim is to determine a cost-effective yet accurate approach for summarizing legislative discussions from the Kosovo Parliament in Albanian.

### Objectives

1. **Fine-Tuning a Language Model:**
   - Implement a fine-tuning approach on a pre-trained model to generate accurate summaries from Albanian texts, assessing its effectiveness and resource implications.
   
2. **Exploring Prompt Engineering Techniques:**
   - Evaluate zero-shot, one-shot, and multi-shot prompt engineering to understand their efficiency and performance without extensive computational resources.

### Comparison Objective

The core objective is to compare fine-tuning and prompt engineering in terms of cost, performance, and practicality, to decide if the investment in fine-tuning is justified or if prompt-based approaches yield comparable outcomes.

![Diagram](images/diagram.jpg "Kosovo Assembly Session")

### Data

1. **Data Source and Preparation:**
   - Original speeches were sourced from the Kosovo Assembly's website, initially in PDF format and converted to text via OCR, introducing potential errors which were later cleaned manually and published in  hugging face: [Kushtrim/Kosovo-Parliament-Transcriptions](https://huggingface.co/datasets/Kushtrim/Kosovo-Parliament-Transcriptions).

2. **Dataset Structure:**
   - The data comprises attributes like `text`, `speaker`, `date`, and `id`, with speeches across multiple languages due to the multilingual nature of the Assembly's proceedings.

## Data Preprocessing

#### Creating Labeled Data

The primary challenge in our project is the absence of labeled data suitable for training a summarization model directly. As our dataset, the "Kosovo-Parliament-Transcriptions," is primarily unlabeled, our initial task involves generating this crucial labeled dataset.

#### Preparing the Data
- **Data Cleaning**: Ensure the cleanliness of the dataset by removing entries with missing or incomplete data
- **Language Detection and Filtering**: Speeches not detected as Albanian (`sq`) are filtered out
- **Token Count and Speech Filtering**: Filter speeches based on token counts to ensure that they are of suitable length for summarization
  - The filtering from 200 to 1024 tokens nsures that the data is neither too sparse for meaningful summarization nor too lengthy for processing efficiency.
- **Data Sampling**: From the filtered dataset, a random sample of 1000 speeches is selected. This sample size is chosen to maintain a diverse range of topics and discussions, ensuring that the model is not biased towards any particular type of speech or session.
- **Translation for Summarization**: Translate Albanian speeches into English to leverage advanced NLP models optimized for English, enhancing model compatibility and improving summary quality.
  - **BLEU Score Evaluation**: Measure translation performance with BLEU scores, using the formula to assess precision and fluency.
    ![BLEU Score Formula](path/to/your/bleu_formula_image.png)
  - Google Translate achieved an average BLEU score of 0.45, surpassing the Helsinki-NLP opus-mt model's 0.30, affirming its higher quality and accuracy for project use.

   
