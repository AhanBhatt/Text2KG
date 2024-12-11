# Text2KG  

This repository contains the implementation for **Text2KG**, a research project focused on automated knowledge graph (KG) generation from unstructured data using three prominent language models: GPT-4, LLaMA 2 (13B), and BERT. The goal is to simplify KG creation for GraphRAGs (Graph-based Retrieval-Augmented Generative Systems) and evaluate the models based on several performance metrics.

## Features  
- **Automated Knowledge Graph Creation:** Generate KGs directly from raw text without manual relationship classification.  
- **Multi-Model Support:** Scripts for GPT-4, LLaMA 2, and BERT are provided.  
- **Performance Evaluation:** Includes metrics such as Precision, Recall, F1-Score, Graph Edit Distance, and Semantic Similarity for comparing the generated KGs.  
- **Dataset:** Contains the Wikipedia excerpt on C programming language used as input data.  
- **Visualization:** Generates visual representations of KGs for all three models.  

## Repository Structure  
```plaintext
.
├── data/                  # Input dataset (Wikipedia excerpt)
├── models/                # Scripts for GPT-4, LLaMA 2, and BERT
│   ├── gpt4.py            # GPT-4 KG generation script
│   ├── llama2.py          # LLaMA 2 KG generation script
│   └── bert.py            # BERT KG generation script
├── metrics/               # Scripts to calculate evaluation metrics
├── visualizations/        # Generated KG visualizations
├── results/               # Output tables and evaluation results
├── docs/                  # Detailed methodology and instructions
├── README.md              # Project overview (this file)
└── requirements.txt       # Dependencies for the project
```
## Usage

1. **Clone the repository**:

    ```bash
    git clone https://github.com/AhanBhatt/Text2KG.git
    cd Text2KG
    ```

2. **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

3. **Run KG generation**:
    - To generate KGs using GPT-4:
        ```bash
        python models/gpt4.py
        ```
    - To generate KGs using LLaMA 2:
        ```bash
        python models/llama2.py
        ```
    - To generate KGs using BERT:
        ```bash
        python models/bert.py
        ```

4. **Visualize and Evaluate**:
    - Visualizations will be saved in `/visualizations/`.
    - Evaluation metrics will be stored in `/results/`.

---

## Key Highlights

- Provides a systematic comparison of GPT-4, LLaMA 2, and BERT for KG creation.
- Reproducible methodology with pre-written scripts for all three models.
- Evaluation focuses on precision, accuracy, and structural fidelity of KGs.

---

## Applications

This project is a step toward automating KG creation for **GraphRAGs** and has applications in:

- Information retrieval
- Knowledge representation
- AI-driven reasoning systems

---

## Contributing

Feel free to contribute by extending the dataset, optimizing model implementations, or suggesting additional evaluation metrics.

---
